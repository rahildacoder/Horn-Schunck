import cv2
import numpy as np
import os

def extract_frames(video_path, output_dir):
    """
    Extract frames from video as grayscale float32 binary files and PNG images.
    """
    # Load video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video info
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height}, {total_frames} frames, {fps:.2f} fps")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    frame_idx = 0
    prev_gray = None

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check if frame is distinct from previous
        if prev_gray is not None:
            diff = np.sum(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
            if diff == 0:
                print(f"Warning: Frame {frame_idx} is identical to frame {frame_idx - 1}")
            else:
                print(f"Frame {frame_idx}: diff from prev = {diff:.0f}")

        prev_gray = gray.copy()

        # Normalize to 0-1 float32
        gray_float = gray.astype(np.float32) / 255.0

        # Save as raw binary (row-major order)
        bin_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.bin")
        gray_float.tofile(bin_path)

        # Save as PNG for visualization
        png_path = os.path.join(png_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(png_path, gray)

        frame_idx += 1

    video.release()

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"{width} {height} {frame_idx} {fps}\n")

    print(f"\nDone! Extracted {frame_idx} frames")
    print(f"Binary files: {output_dir}/frame_XXXX.bin")
    print(f"PNG files: {png_dir}/frame_XXXX.png")
    print(f"Metadata: {metadata_path}")
    print(f"Frame size: {width}x{height} = {width * height} pixels")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "input.mp4")
    output_dir = os.path.join(script_dir, "frames")

    extract_frames(video_path, output_dir)
