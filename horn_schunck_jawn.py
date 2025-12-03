import cv2
import numpy as np
from scipy import ndimage
import os
import subprocess
import platform

class HornSchunck:
    def __init__(self, alpha=1.0, iterations=100):
        self.alpha = alpha
        self.iterations = iterations
    
    def compute_flow(self, img1, img2):
        """
        Compute Horn-Schunck optical flow between two consecutive frames
        """
        # Convert to float and normalize
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        # Image dimensions
        h, w = img1.shape
        
        # Initialize flow arrays
        u = np.zeros((h, w))
        v = np.zeros((h, w))
        
        # Compute image derivatives
        Ix = ndimage.sobel(img1, axis=1, mode='constant')
        Iy = ndimage.sobel(img1, axis=0, mode='constant')
        It = img2 - img1
        
        # Kernel for averaging neighbors
        kernel = np.array([[1/12, 1/6, 1/12],
                          [1/6,    0, 1/6],
                          [1/12, 1/6, 1/12]], dtype=np.float32)
        
        # Iteratively solve for optical flow
        for _ in range(self.iterations):
            # Compute local averages of flow vectors
            u_avg = cv2.filter2D(u, -1, kernel)
            v_avg = cv2.filter2D(v, -1, kernel)
            
            # Update flow vectors
            denominator = self.alpha**2 + Ix**2 + Iy**2
            u = u_avg - Ix * (Ix * u_avg + Iy * v_avg + It) / denominator
            v = v_avg - Iy * (Ix * u_avg + Iy * v_avg + It) / denominator
        
        return u, v

def get_compatible_codec():
    """Get a compatible video codec for the current system"""
    codecs_to_try = []
    
    if platform.system() == "Windows":
        codecs_to_try = ['DIVX', 'XVID', 'MJPG', 'MP4V']
    elif platform.system() == "Darwin":  # macOS
        codecs_to_try = ['avc1', 'mp4v', 'XVID', 'MJPG']
    else:  # Linux
        codecs_to_try = ['XVID', 'MJPG', 'MP4V', 'DIVX']
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            # Test by creating a temporary writer
            test_writer = cv2.VideoWriter('test_temp.avi', fourcc, 30, (640, 480))
            if test_writer.isOpened():
                test_writer.release()
                # Clean up test file if it exists
                if os.path.exists('test_temp.avi'):
                    os.remove('test_temp.avi')
                print(f"Using codec: {codec}")
                return fourcc
        except:
            continue
    
    print("Warning: No optimal codec found, using XVID as fallback")
    return cv2.VideoWriter_fourcc(*'XVID')

def create_sample_video():
    """Create a simple sample video for testing"""
    print("Creating sample video...")
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds
    total_frames = fps * duration
    
    output_file = "sample_input.mp4"
    fourcc = get_compatible_codec()
    
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not create sample video file")
        return None
    
    for i in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray background
        
        # Create a moving rectangle
        rect_size = 80
        x_pos = (i * 8) % (width - rect_size)
        y_pos = (height - rect_size) // 2 + int(50 * np.sin(i * 0.2))  # Sine wave motion
        
        # Draw moving rectangle
        cv2.rectangle(frame, (x_pos, y_pos), 
                     (x_pos + rect_size, y_pos + rect_size), 
                     (0, 200, 0), -1)
        
        # Draw a second moving circle
        circle_radius = 40
        circle_x = (width // 4 + (i * 6) % (width // 2))
        circle_y = height // 4 + int(30 * np.cos(i * 0.15))
        
        cv2.circle(frame, (circle_x, circle_y), circle_radius, (200, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, "Optical Flow Test Video", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Sample video created: {output_file}")
    return output_file

def process_video_with_optical_flow(input_path, output_path, alpha=1.0, iterations=100, 
                                   scale_factor=5, threshold=0.5, step_size=8):
    """
    Process video with Horn-Schunck optical flow overlay
    """
    
    # Initialize Horn-Schunck optical flow
    hs = HornSchunck(alpha=alpha, iterations=iterations)
    
    # Open video capture
    cap = cv2.VideoCapture(input_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default if cannot read FPS
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Input video: {width}x{height} at {fps} FPS")
    
    # Use compatible codec
    fourcc = get_compatible_codec()
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video '{output_path}'")
        print("Trying with MP4V codec...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            return False
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame from video")
        cap.release()
        out.release()
        return False
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_count = 0
    print("Processing video with Horn-Schunck optical flow...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow
        u, v = hs.compute_flow(prev_gray, gray)
        
        # Create output frame with overlay
        output_frame = frame.copy()
        
        # Create heatmap for motion magnitude
        magnitude = np.sqrt(u**2 + v**2)
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(magnitude_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend heatmap with original frame
        alpha_blend = 0.3
        output_frame = cv2.addWeighted(output_frame, 1 - alpha_blend, heatmap, alpha_blend, 0)
        
        # Draw flow vectors
        h, w = gray.shape
        y, x = np.mgrid[step_size//2:h:step_size, step_size//2:w:step_size].reshape(2, -1).astype(int)
        
        vectors_drawn = 0
        for i, (xi, yi) in enumerate(zip(x, y)):
            if xi < w and yi < h:
                dx = u[yi, xi]
                dy = v[yi, xi]
                mag = np.sqrt(dx**2 + dy**2)
                
                if mag > threshold:
                    # Calculate endpoints
                    x2 = int(xi + dx * scale_factor)
                    y2 = int(yi + dy * scale_factor)
                    
                    # Only draw if the endpoint is within frame bounds
                    if 0 <= x2 < w and 0 <= y2 < h:
                        # Draw arrow
                        cv2.arrowedLine(output_frame, (xi, yi), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
                        vectors_drawn += 1
        
        # Add text information
        cv2.putText(output_frame, f"Horn-Schunck Optical Flow", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Frame: {frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Motion Vectors: {vectors_drawn}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame to output video
        out.write(output_frame)
        
        # Update previous frame
        prev_gray = gray.copy()
        frame_count += 1
        
        # Display progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete! Output saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")
    
    # Verify the output file was created and can be opened
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Output file size: {file_size / (1024*1024):.2f} MB")
        
        # Try to open the output video to verify it's valid
        cap_test = cv2.VideoCapture(output_path)
        if cap_test.isOpened():
            test_frames = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_test.release()
            print(f"Output video verified: {test_frames} frames")
        else:
            print("Warning: Output video may not be playable with OpenCV")
    else:
        print("Error: Output file was not created")
        return False
    
    return True

def open_video_file(file_path):
    """Try to open the video file with system default player"""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist")
        return False
    
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(['open', file_path])
        else:  # Linux
            subprocess.run(['xdg-open', file_path])
        print(f"Attempted to open: {file_path}")
        return True
    except Exception as e:
        print(f"Could not open video automatically: {e}")
        print(f"Please open the file manually: {file_path}")
        return False

def realtime_optical_flow():
    """Real-time optical flow from webcam"""
    print("Starting real-time optical flow from webcam...")
    print("Press 'q' to quit, 'r' to reset background")
    
    hs = HornSchunck(alpha=1.0, iterations=50)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow
        u, v = hs.compute_flow(prev_gray, gray)
        
        # Create visualization
        magnitude = np.sqrt(u**2 + v**2)
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(magnitude_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original frame
        output_frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
        
        # Draw flow vectors
        h, w = gray.shape
        step = 20
        vectors_drawn = 0
        
        for y in range(step//2, h, step):
            for x in range(step//2, w, step):
                dx = u[y, x]
                dy = v[y, x]
                mag = np.sqrt(dx**2 + dy**2)
                
                if mag > 1.0:
                    x2 = int(x + dx * 5)
                    y2 = int(y + dy * 5)
                    if 0 <= x2 < w and 0 <= y2 < h:
                        cv2.arrowedLine(output_frame, (x, y), (x2, y2), (0, 255, 0), 2, tipLength=0.3)
                        vectors_drawn += 1
        
        # Add info text
        cv2.putText(output_frame, f"Real-time Optical Flow - Vectors: {vectors_drawn}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, "Press 'q' to quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Horn-Schunck Optical Flow', output_frame)
        
        prev_gray = gray.copy()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset background
            prev_gray = gray.copy()
            print("Background reset")
    
    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    print("Horn-Schunck Optical Flow Demo")
    print("=" * 40)
    
    while True:
        print("\nChoose an option:")
        print("1. Process video file")
        print("2. Create sample video and process it")
        print("3. Real-time webcam optical flow")
        print("4. Open output video file")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            input_file = input("Enter input video file path: ").strip()
            output_file = input("Enter output video file path [default: output.mp4]: ").strip()
            if not output_file:
                output_file = "output.mp4"
            
            if os.path.exists(input_file):
                success = process_video_with_optical_flow(input_file, output_file)
                if success:
                    print("Would you like to open the output video now? (y/n)")
                    if input().strip().lower() == 'y':
                        open_video_file(output_file)
            else:
                print(f"Error: File '{input_file}' not found")
        
        elif choice == '2':
            sample_file = create_sample_video()
            if sample_file:
                output_file = "output_with_optical_flow.mp4"
                success = process_video_with_optical_flow(sample_file, output_file)
                if success:
                    print("Would you like to open the output video now? (y/n)")
                    if input().strip().lower() == 'y':
                        open_video_file(output_file)
        
        elif choice == '3':
            realtime_optical_flow()
        
        elif choice == '4':
            video_file = input("Enter video file path to open: ").strip()
            open_video_file(video_file)
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")