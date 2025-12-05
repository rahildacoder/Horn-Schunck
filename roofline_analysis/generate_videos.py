#!/usr/bin/env python3
"""
Generate videos at different resolutions for roofline analysis.
Uses OpenCV for video processing.
"""

import cv2
import os
import sys

# Define resolutions to test
resolutions = [
    ("360p", 640, 360),
    ("720p", 1280, 720),
    ("1080p", 1920, 1080),
    ("1440p", 2560, 1440),
    ("2160p", 3840, 2160),
]

def generate_video(input_file, output_file, width, height):
    """Generate a video at the specified resolution using OpenCV"""
    print(f"Generating {output_file} at {width}x{height}...")

    # Open input video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"✗ Failed to open {input_file}")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    # Create output video writer
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"✗ Failed to create {output_file}")
        cap.release()
        return False

    # Process frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        out.write(resized)
        frame_count += 1

    # Cleanup
    cap.release()
    out.release()

    print(f"✓ Created {output_file} ({frame_count} frames)")
    return True

def main():
    # Check if input.mp4 exists
    if not os.path.exists("input2.mp4"):
        print("Error: input.mp4 not found in current directory")
        sys.exit(1)

    print("="*60)
    print("Generating videos at different resolutions using OpenCV")
    print("="*60)

    # Generate videos at each resolution
    success_count = 0
    for name, width, height in resolutions:
        output_file = f"input2_{name}.mp4"
        if generate_video("input2.mp4", output_file, width, height):
            success_count += 1
        print()

    print("="*60)
    print(f"Generated {success_count}/{len(resolutions)} videos")
    print("="*60)

    # List generated files
    print("\nGenerated files:")
    for name, _, _ in resolutions:
        filename = f"input2_{name}.mp4"
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"  {filename:20s} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
