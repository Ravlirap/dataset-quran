import os
import re
import json
import wave
import struct
import argparse
import whisper

from Levenshtein import ratio

DATASET_DIR = "dataset"
METADATA_FILE = os.path.join(DATASET_DIR, "metadata.json")


# ==========================
# NORMALISASI ARAB
# ==========================

def normalize_arabic(text):

    if not text:
        return ""

    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[\u06D6-\u06ED]', '', text)
    text = text.replace('\u0640', '')

    text = re.sub(
        r'[\u0622\u0623\u0625]',
        '\u0627',
        text
    )

    text = text.replace('\u0629', '\u0647')
    text = text.replace('\u0649', '\u064A')

    text = re.sub(
        r'[^\u0621-\u063A\u0641-\u064A\s]',
        '',
        text
    )

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ==========================
# AUDIO CHECK
# ==========================

def check_audio(path):

    result = {
        "valid": True,
        "issues": []
    }

    try:

        with wave.open(path, "rb") as wf:

            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            sampwidth = wf.getsampwidth()

            duration = frames / sample_rate

            raw = wf.readframes(frames)

        if sample_rate != 16000:
            result["issues"].append(
                f"SampleRate={sample_rate}"
            )

        if channels != 1:
            result["issues"].append(
                f"Channels={channels}"
            )

        if duration < 0.3:
            result["issues"].append(
                "TooShort"
            )

        if sampwidth == 2:

            samples = struct.unpack(
                f"<{frames * channels}h",
                raw[:frames * channels * 2]
            )

            peak = max(abs(x) for x in samples)

            if peak > 32000:
                result["issues"].append(
                    "PossibleClipping"
                )

        result["duration"] = round(duration, 2)

    except Exception as e:

        result["valid"] = False
        result["issues"].append(str(e))

    return result


# ==========================
# SPEAKER FILTER
# ==========================

def filter_speaker_range(
        metadata,
        start_id,
        end_id):

    filtered = []

    for item in metadata:

        sid = int(
            item["speaker_id"]
            .split("_")[1]
        )

        if start_id <= sid <= end_id:
            filtered.append(item)

    return filtered


# ==========================
# MAIN
# ==========================

def run_validation(
        start_speaker,
        end_speaker,
        model_name):

    print()
    print("=" * 60)
    print("MAUBAIK DATASET VALIDATOR")
    print("=" * 60)

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    metadata = filter_speaker_range(
        metadata,
        start_speaker,
        end_speaker
    )

    print(
        f"Speaker: "
        f"{start_speaker:03d}"
        f" - "
        f"{end_speaker:03d}"
    )

    print(
        f"Audio : {len(metadata)}"
    )

    print()
    print(
        "Loading Whisper..."
    )

    model = whisper.load_model(
        model_name
    )

    audio_ok = 0

    excellent = 0
    good = 0
    review = 0
    bad = 0

    similarities = []

    details = []

    for item in metadata:

        audio_rel = item["audio"]

        audio_path = os.path.join(
            DATASET_DIR,
            audio_rel
        )

        if not os.path.exists(audio_path):

            details.append({
                "audio": audio_rel,
                "status": "MISSING"
            })

            continue

        quality = check_audio(
            audio_path
        )

        if len(quality["issues"]) == 0:
            audio_ok += 1

        try:

            result = model.transcribe(
                audio_path,
                language="ar"
            )

            predicted = normalize_arabic(
                result["text"]
            )

            expected = normalize_arabic(
                item["text"]
            )

            sim = ratio(
                expected,
                predicted
            )

        except Exception:

            sim = 0

        similarities.append(sim)

        if sim >= 0.85:
            excellent += 1
            status = "EXCELLENT"

        elif sim >= 0.70:
            good += 1
            status = "GOOD"

        elif sim >= 0.50:
            review += 1
            status = "REVIEW"

        else:
            bad += 1
            status = "BAD"

        details.append({

            "speaker_id":
                item["speaker_id"],

            "audio":
                audio_rel,

            "similarity":
                round(sim * 100, 2),

            "status":
                status,

            "issues":
                quality["issues"]
        })

    avg = 0

    if similarities:
        avg = (
            sum(similarities)
            / len(similarities)
        ) * 100

    print()
    print("=" * 60)

    print(
        f"Audio Valid : "
        f"{audio_ok}/{len(metadata)}"
    )

    print(
        f"Excellent : {excellent}"
    )

    print(
        f"Good      : {good}"
    )

    print(
        f"Review    : {review}"
    )

    print(
        f"Bad       : {bad}"
    )

    print(
        f"Average Similarity : "
        f"{avg:.2f}%"
    )

    print("=" * 60)

    report = {

        "summary": {

            "speaker_range":
                f"{start_speaker:03d}-{end_speaker:03d}",

            "total_audio":
                len(metadata),

            "audio_valid":
                audio_ok,

            "excellent":
                excellent,

            "good":
                good,

            "review":
                review,

            "bad":
                bad,

            "avg_similarity":
                round(avg, 2)
        },

        "details":
            details
    }

    report_name = (
        f"report_"
        f"{start_speaker:03d}_"
        f"{end_speaker:03d}.json"
    )

    with open(
        report_name,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nReport saved: "
        f"{report_name}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        type=int,
        required=True
    )

    parser.add_argument(
        "--end",
        type=int,
        required=True
    )

    parser.add_argument(
        "--model",
        default="small"
    )

    args = parser.parse_args()

    run_validation(
        args.start,
        args.end,
        args.model
    )