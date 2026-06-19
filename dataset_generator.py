"""
dataset_generator.py — build the synthetic training set.

Compiles the C++/CUDA pipeline, runs it in --generate-dataset mode to populate
RAW_DATA/ with 16-frame clips (IR density frames + ground-truth labels), and
reports what was produced. Fully local, no network, no external services.
"""

import os
import glob
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def compile_and_run(num_clips=10):
    print("[NIGHTWATCH-SYNTH] Building the C++/CUDA pipeline...")
    # Rebuild to pick up any changes in main.cpp / kernels.
    subprocess.run(["build.bat"], shell=True)

    print(f"\nGenerating {num_clips} synthetic clips (16 frames each)...")
    result = subprocess.run(
        ["nightwatch_vision.exe", "--generate-dataset", str(num_clips)],
        capture_output=True, text=True,
    )

    if "ERROR" in result.stdout or result.returncode != 0:
        print("Generation failed:")
        print(result.stdout)
        raise SystemExit(1)
    print("OK: generation complete (C++ / CUDA).\n")


def analyze_dataset():
    clips = glob.glob("RAW_DATA/clip_*")
    print(f"Found {len(clips)} clips in RAW_DATA.")

    if len(clips) > 0:
        sample_clip = clips[0]
        frames = glob.glob(f"{sample_clip}/frame_*.bin")
        gts    = glob.glob(f"{sample_clip}/gt_*.txt")
        print(f"   -> sample clip '{sample_clip}': {len(frames)} frames, {len(gts)} labels.")
        return len(clips), len(frames)
    return 0, 0


if __name__ == "__main__":
    num_clips_to_generate = 5  # 5 clips * 16 = 80 frames
    compile_and_run(num_clips_to_generate)
    clips_count, frames_count = analyze_dataset()
    print(f"\nDataset ready for PyTorch: {clips_count} clips, "
          f"{frames_count} frames/clip. Train with: python train.py")
