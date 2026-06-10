import os
import io
import json
import re
import wave
import struct
import math
import tempfile
import argparse

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATASET_DIR    = "dataset"
METADATA_FILE  = "dataset/metadata.json"
OPENAI_API_KEY = ""         

# Threshold silence trimming
SILENCE_THRESHOLD = 0.015    # amplitudo di bawah ini dianggap sunyi
SILENCE_PAD_MS    = 100      # sisakan 100ms di awal & akhir (natural)


# ─────────────────────────────────────────────
# 1. NORMALISASI TEKS ARAB
# ─────────────────────────────────────────────
def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)   # harakat
    text = re.sub(r'[\u06D6-\u06ED]', '', text)          # tanda baca Quran
    text = text.replace('\u0640', '')                     # tatweel
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)  # alef
    text = text.replace('\u0629', '\u0647')               # ta marbuta
    text = text.replace('\u0649', '\u064A')               # ya
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────
# 2. LEVENSHTEIN SIMILARITY
# ─────────────────────────────────────────────
def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def similarity_score(ref: str, hyp: str) -> float:
    nr, nh = normalize_arabic(ref), normalize_arabic(hyp)
    if not nr and not nh:
        return 1.0
    if not nr or not nh:
        return 0.0
    dist = levenshtein_distance(nr, nh)
    return round(1.0 - dist / max(len(nr), len(nh)), 4)


# ─────────────────────────────────────────────
# 3. SILENCE TRIMMING (di memory, file asli TIDAK berubah)
# ─────────────────────────────────────────────
def trim_silence_in_memory(wav_path: str) -> bytes | None:
    """
    Baca WAV, potong silence di awal & akhir di memory,
    kembalikan bytes WAV yang sudah ditrim.
    File asli di disk sama sekali tidak disentuh.
    Kembalikan None jika file tidak valid / tidak bisa dibaca.
    """
    try:
        with wave.open(wav_path, 'rb') as wf:
            channels   = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            n_frames   = wf.getnframes()
            raw        = wf.readframes(n_frames)

        # Parse ke list sampel int
        if sampwidth == 2:
            fmt     = f"<{n_frames * channels}h"
            samples = list(struct.unpack(fmt, raw[:n_frames * channels * 2]))
            max_val = 32768.0
        elif sampwidth == 1:
            fmt     = f"{n_frames * channels}B"
            samples = [s - 128 for s in struct.unpack(fmt, raw)]
            max_val = 128.0
        else:
            return None   # format tidak didukung

        # Jika stereo, ambil rata-rata tiap frame untuk deteksi silence
        if channels == 2:
            mono_check = [(abs(samples[i]) + abs(samples[i+1])) / 2
                          for i in range(0, len(samples), 2)]
        else:
            mono_check = [abs(s) for s in samples]

        threshold  = SILENCE_THRESHOLD * max_val
        pad_frames = int(framerate * SILENCE_PAD_MS / 1000)

        # Cari frame pertama yang di atas threshold
        start_frame = 0
        for i, amp in enumerate(mono_check):
            if amp > threshold:
                start_frame = max(0, i - pad_frames)
                break

        # Cari frame terakhir yang di atas threshold
        end_frame = len(mono_check)
        for i in range(len(mono_check) - 1, -1, -1):
            if mono_check[i] > threshold:
                end_frame = min(len(mono_check), i + pad_frames + 1)
                break

        # Kalau semua sunyi (silence total), kembalikan original
        if start_frame >= end_frame:
            return None

        # Potong samples sesuai frame range
        if channels == 2:
            trimmed = samples[start_frame * 2 : end_frame * 2]
        else:
            trimmed = samples[start_frame:end_frame]

        # Tulis kembali ke bytes WAV di memory (tidak ke disk)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wout:
            wout.setnchannels(channels)
            wout.setsampwidth(sampwidth)
            wout.setframerate(framerate)
            if sampwidth == 2:
                wout.writeframes(struct.pack(f"<{len(trimmed)}h", *trimmed))
            else:
                wout.writeframes(struct.pack(f"{len(trimmed)}B",
                                             *[s + 128 for s in trimmed]))

        trimmed_bytes = buf.getvalue()

        # Info berapa detik yang dipotong
        original_sec = n_frames / framerate
        trimmed_sec  = (end_frame - start_frame) / framerate
        cut_sec      = original_sec - trimmed_sec

        return trimmed_bytes, round(cut_sec, 2)

    except Exception:
        return None


# ─────────────────────────────────────────────
# 4. CEK KUALITAS AUDIO
# ─────────────────────────────────────────────
def check_audio_quality(wav_path: str) -> dict:
    result = {
        "exists": False, "is_wav": False,
        "sample_rate": None, "channels": None,
        "duration_sec": None, "max_amplitude": None,
        "rms_amplitude": None, "is_clipping": False,
        "is_too_quiet": False, "has_long_silence": False,
        "issues": [], "passed": False,
    }

    full_path = os.path.join(DATASET_DIR, wav_path)
    if not os.path.exists(full_path):
        result["issues"].append("FILE TIDAK ADA")
        return result

    result["exists"] = True
    if not wav_path.lower().endswith(".wav"):
        result["issues"].append("Bukan file WAV")
        return result
    result["is_wav"] = True

    try:
        with wave.open(full_path, 'rb') as wf:
            channels   = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames   = wf.getnframes()
            sampwidth  = wf.getsampwidth()
            duration   = n_frames / sample_rate
            raw        = wf.readframes(n_frames)

        result.update({"sample_rate": sample_rate, "channels": channels,
                        "duration_sec": round(duration, 2)})

        if sampwidth == 2:
            samples = list(struct.unpack(f"<{n_frames * channels}h", raw[:n_frames * channels * 2]))
            max_val = 32768.0
        elif sampwidth == 1:
            samples = [s - 128 for s in struct.unpack(f"{n_frames * channels}B", raw)]
            max_val = 128.0
        else:
            samples = []
            max_val = 1.0

        if samples:
            abs_s = [abs(s) for s in samples]
            max_amp = max(abs_s) / max_val
            rms     = math.sqrt(sum(s*s for s in samples) / len(samples)) / max_val
            result.update({"max_amplitude": round(max_amp, 4),
                            "rms_amplitude": round(rms, 4),
                            "is_clipping":   max_amp >= 0.98,
                            "is_too_quiet":  rms < 0.01})
            head = abs_s[:min(int(sample_rate * 0.5) * channels, len(abs_s))]
            result["has_long_silence"] = (
                math.sqrt(sum(s*s for s in head) / len(head)) / max_val < 0.005
                if head else False
            )

        if sample_rate != 16000: result["issues"].append(f"Sample rate {sample_rate} Hz (harus 16000)")
        if channels != 1:        result["issues"].append(f"Audio {channels} ch (harus mono)")
        if duration < 0.3:       result["issues"].append(f"Terlalu pendek ({duration:.1f}s)")
        if duration > 30:        result["issues"].append(f"Terlalu panjang ({duration:.1f}s)")
        if result["is_clipping"]:     result["issues"].append("Clipping / pecah")
        if result["is_too_quiet"]:    result["issues"].append("Terlalu pelan (RMS < 1%)")
        if result["has_long_silence"]:result["issues"].append("Silence panjang di awal (akan ditrim)")

    except Exception as e:
        result["issues"].append(f"Error baca WAV: {e}")

    result["passed"] = len(result["issues"]) == 0 or result["issues"] == ["Silence panjang di awal (akan ditrim)"]
    return result


# ─────────────────────────────────────────────
# 5. TRANSKRIPSI
# ─────────────────────────────────────────────
def transcribe_local(wav_path: str, trimmed_bytes: bytes | None,
                     model_size="base") -> str:
    try:
        import whisper
        model     = whisper.load_model(model_size)
        full_path = os.path.join(DATASET_DIR, wav_path)

        if trimmed_bytes:
            # Tulis bytes ke file temp sementara, lalu hapus setelah selesai
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(trimmed_bytes)
                tmp_path = tmp.name
            try:
                result = model.transcribe(tmp_path, language="ar")
            finally:
                os.unlink(tmp_path)   # hapus file temp
        else:
            result = model.transcribe(full_path, language="ar")

        return result["text"].strip()
    except ImportError:
        print("  [ERROR] Jalankan: pip install openai-whisper")
        return ""
    except Exception as e:
        print(f"  [ERROR] Transkripsi gagal: {e}")
        return ""


def transcribe_api(wav_path: str, trimmed_bytes: bytes | None) -> str:
    try:
        from openai import OpenAI
        client    = OpenAI(api_key=OPENAI_API_KEY)
        full_path = os.path.join(DATASET_DIR, wav_path)

        if trimmed_bytes:
            audio_file = ("audio.wav", io.BytesIO(trimmed_bytes), "audio/wav")
        else:
            audio_file = open(full_path, "rb")

        result = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, language="ar"
        )
        if not trimmed_bytes:
            audio_file.close()
        return result.text.strip()
    except ImportError:
        print("  [ERROR] Jalankan: pip install openai")
        return ""
    except Exception as e:
        print(f"  [ERROR] API gagal: {e}")
        return ""


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_test(mode="local", sample=None, speaker_filter=None,
             whisper_model="base", use_trim=True):

    print("=" * 60)
    print("  MAUBAIK DATASET TESTER")
    print(f"  Mode       : {'Whisper Lokal' if mode == 'local' else 'Whisper API'}")
    print(f"  Silence trim: {'AKTIF (di memory, file asli tidak berubah)' if use_trim else 'NONAKTIF'}")
    print("=" * 60)

    if not os.path.exists(METADATA_FILE):
        print(f"\n[ERROR] {METADATA_FILE} tidak ditemukan!")
        return

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"\n[INFO] Total entri di metadata: {len(metadata)}")

    if speaker_filter:
        metadata = [e for e in metadata if e["speaker_id"] == speaker_filter]
        print(f"[INFO] Filter speaker '{speaker_filter}': {len(metadata)} entri")

    if sample:
        metadata = metadata[:sample]
        print(f"[INFO] Mode sample: test {sample} audio pertama")

    print()

    results   = []
    total     = len(metadata)
    total_sim = 0.0
    passed_audio = 0
    total_cut_sec = 0.0

    for idx, entry in enumerate(metadata):
        audio    = entry["audio"]
        ref_text = entry["text"]

        print(f"[{idx+1:3d}/{total}] {audio}")

        # --- Cek kualitas ---
        quality = check_audio_quality(audio)

        if not quality["exists"]:
            print(f"         ✗ FILE TIDAK ADA\n")
            results.append({**entry, "quality": quality,
                             "transcription": "", "similarity": 0.0, "status": "MISSING"})
            continue

        q_issues = [i for i in quality["issues"] if "Silence" not in i]
        q_ok = "✓ OK" if not q_issues else f"⚠ {', '.join(q_issues)}"
        print(f"         Audio  : {q_ok}")
        if not q_issues:
            passed_audio += 1

        # --- Silence trimming di memory ---
        trimmed_bytes = None
        cut_sec = 0.0
        if use_trim:
            full_path = os.path.join(DATASET_DIR, audio)
            trim_result = trim_silence_in_memory(full_path)
            if trim_result:
                trimmed_bytes, cut_sec = trim_result
                total_cut_sec += cut_sec
                if cut_sec > 0.05:
                    print(f"         Trim   : dipotong {cut_sec}s silence (file asli aman)")

        # --- Transkripsi ---
        if mode == "local":
            transcription = transcribe_local(audio, trimmed_bytes, whisper_model)
        else:
            transcription = transcribe_api(audio, trimmed_bytes)

        # --- Similarity ---
        sim     = similarity_score(ref_text, transcription) if transcription else 0.0
        total_sim += sim
        sim_pct   = round(sim * 100, 1)
        status    = "PASS" if sim >= 0.80 else ("WARN" if sim >= 0.50 else "FAIL")
        label     = {"PASS": "✓ PASS", "WARN": "⚠ PERLU PERBAIKAN", "FAIL": "✗ FAIL"}[status]

        print(f"         Transkrip: {transcription[:60] if transcription else '(kosong)'}")
        print(f"         Referensi: {ref_text[:60]}")
        print(f"         Similarity: {sim_pct}%  {label}")
        print()

        results.append({
            **entry,
            "quality":       quality,
            "transcription": transcription,
            "similarity":    sim,
            "similarity_pct": sim_pct,
            "cut_silence_sec": cut_sec,
            "status":        status,
        })

    # ─── Ringkasan ───
    tested     = len([r for r in results if r["status"] != "MISSING"])
    passed_sim = len([r for r in results if r["status"] == "PASS"])
    warned_sim = len([r for r in results if r["status"] == "WARN"])
    failed_sim = len([r for r in results if r["status"] == "FAIL"])
    missing    = len([r for r in results if r["status"] == "MISSING"])
    avg_sim    = round(total_sim / tested * 100, 1) if tested else 0

    print("=" * 60)
    print("  HASIL PENGUJIAN")
    print("=" * 60)
    print(f"  Total entri diuji      : {total}")
    print(f"  File ditemukan         : {tested}")
    print(f"  File tidak ada         : {missing}")
    print(f"  Audio lolos kualitas   : {passed_audio}/{tested}")
    if use_trim:
        print(f"  Total silence dipotong : {round(total_cut_sec, 1)}s (di memory)")
    print(f"  Similarity ≥ 80% (PASS) : {passed_sim}")
    print(f"  Similarity 50–79% (WARN): {warned_sim}")
    print(f"  Similarity < 50%  (FAIL): {failed_sim}")
    print(f"  Rata-rata similarity    : {avg_sim}%")
    print("=" * 60)

    if avg_sim >= 85:
        print("  🟢 Dataset sangat baik — siap dikumpulkan!")
    elif avg_sim >= 70:
        print("  🟡 Cukup baik — perbaiki audio yang FAIL")
    elif avg_sim >= 50:
        print("  🟠 Perlu banyak perbaikan sebelum dikumpulkan")
    else:
        print("  🔴 Belum siap — banyak audio bermasalah")
    print()

    # Simpan laporan
    report = {
        "summary": {
            "total": total, "tested": tested, "missing": missing,
            "passed_audio_quality": passed_audio,
            "passed_similarity": passed_sim,
            "warned_similarity": warned_sim,
            "failed_similarity": failed_sim,
            "avg_similarity_pct": avg_sim,
            "silence_trim_mode": "in-memory only" if use_trim else "disabled",
        },
        "details": [
            {
                "speaker_id":      r["speaker_id"],
                "surah":           r["surah"],
                "ayat":            r["ayat"],
                "audio":           r["audio"],
                "similarity_pct":  r.get("similarity_pct", 0),
                "status":          r.get("status"),
                "audio_issues":    r["quality"]["issues"],
                "cut_silence_sec": r.get("cut_silence_sec", 0),
                "transcription":   r.get("transcription", ""),
            }
            for r in results
        ]
    }

    with open("test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Laporan disimpan: test_report.json")

    # Tampilkan audio yang gagal
    failed = [r for r in results if r.get("status") in ("FAIL", "MISSING")]
    if failed:
        print(f"\n  Audio yang perlu diperbaiki ({len(failed)} file):")
        for r in failed[:15]:
            print(f"    - {r['audio']}  [{r['status']}]")
            for issue in r["quality"]["issues"]:
                print(f"        ↳ {issue}")
        if len(failed) > 15:
            print(f"    ... dan {len(failed)-15} lainnya (lihat test_report.json)")
    print()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MauBaik Dataset Tester")
    parser.add_argument("--mode",     default="local", choices=["local", "api"])
    parser.add_argument("--sample",   type=int, default=None)
    parser.add_argument("--speaker",  type=str, default=None)
    parser.add_argument("--model",    default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--no-trim",  action="store_true",
                        help="Nonaktifkan silence trimming")
    args = parser.parse_args()

    if args.mode == "api" and not OPENAI_API_KEY:
        print("[ERROR] Isi OPENAI_API_KEY di bagian atas script!")
    else:
        run_test(
            mode=args.mode,
            sample=args.sample,
            speaker_filter=args.speaker,
            whisper_model=args.model,
            use_trim=not args.no_trim,
        )