#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <sys/types.h>

static float alpha = 1.0f;

// function to output grayscale bin files from input video and return number of files created
int video_to_grayscale_bins(const char* input_video, const char* output_folder) {
    mkdir(output_folder, 0777);

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        printf("Error opening video file\n");
        return 0;
    }

    int frame_count = 0;
    cv::Mat frame, gray_frame;
    while (cap.read(frame)) {
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        gray_frame.convertTo(gray_frame, CV_32F, 1.0 / 255.0);

        char filename[256];
        snprintf(filename, sizeof(filename), "%s/frame_%04d.bin", output_folder, frame_count);

        FILE* file = fopen(filename, "wb");
        if (!file) {
            printf("Error opening %s for write\n", filename);
            break;
        }
        fwrite(gray_frame.ptr<float>(), sizeof(float), gray_frame.rows * gray_frame.cols, file);
        fclose(file);

        frame_count++;
    }
    cap.release();

    printf("Actually extracted %d frames.\n", frame_count);
    return frame_count;
}


// constant function that takes in the input.mp4 video and outputs the number of frames, width, and height
void get_video_properties(const char* filename,int& width, int& height) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        printf("Error opening video file\n");
        return;
    }
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cap.release();
}

std::vector<float> horn_schunck(const std::vector<float>& I1, const std::vector<float>& I2, int width, int height, float alpha) {
    std::vector<float> U(width * height, 0.0f);
    std::vector<float> V(width * height, 0.0f);
    std::vector<float> Ix(width * height, 0.0f);
    std::vector<float> Iy(width * height, 0.0f);
    std::vector<float> It(width * height, 0.0f);

    int num_iterations = 20;

    // Compute image gradients (Ix, Iy, It) which are derivatives in x, y and time
    #pragma omp parallel for
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            Ix[idx] = (I1[idx + 1] - I1[idx - 1] + I2[idx + 1] - I2[idx - 1]) * 0.25f;
            Iy[idx] = (I1[idx + width] - I1[idx - width] + I2[idx + width] - I2[idx - width]) * 0.25f;
            It[idx] = (I2[idx] - I1[idx]);
        }
    }

    std::vector<float> U_avg(width * height, 0.0f);
    std::vector<float> V_avg(width * height, 0.0f);

    // Iterative refinement of flow estimates
    for (int iter = 0; iter < num_iterations; ++iter) {

        // Compute average flow
        #pragma omp parallel for
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int idx = y * width + x;
                U_avg[idx] = (U[idx - 1] + U[idx + 1] + U[idx - width] + U[idx + width]) * 0.25f;
                V_avg[idx] = (V[idx - 1] + V[idx + 1] + V[idx - width] + V[idx + width]) * 0.25f;
            }
        }

        // Update flow estimates
        #pragma omp parallel for
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int idx = y * width + x;
                float numerator = Ix[idx] * U_avg[idx] + Iy[idx] * V_avg[idx] + It[idx];
                float denominator = alpha * alpha + Ix[idx] * Ix[idx] + Iy[idx] * Iy[idx];
                U[idx] = U_avg[idx] - (Ix[idx] * numerator) / denominator;
                V[idx] = V_avg[idx] - (Iy[idx] * numerator) / denominator;
            }
        }
    }

    // Combine U and V into a single output vector
    std::vector<float> flow(width * height * 2, 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < width * height; ++i) {
        flow[2 * i] = U[i];
        flow[2 * i + 1] = V[i];
    }

    return flow;
}

// Helper function to load a frame from a binary file
std::vector<float> load_frame(const char* filename, int width, int height) {
    std::vector<float> frame(width * height);
    FILE* file = fopen(filename, "rb");
    fread(frame.data(), sizeof(float), width * height, file);
    fclose(file);
    return frame;
}

// Helper function to generate zero-padded filenames
std::string padded(int number) {
    std::ostringstream ss;
    ss << std::setw(4) << std::setfill('0') << number;
    return ss.str();
}


int main() {
    int width, height;
    
    get_video_properties("input.mp4", width, height);

    int num_frames = video_to_grayscale_bins("input.mp4", "frames");

    printf("Video properties - Frames: %d, Width: %d, Height: %d\n", num_frames, width, height);

    std::vector<std::vector<float>> frames(num_frames);

    for (int i = 0; i < num_frames; ++i) {
        std::string filename = "frames/frame_" + padded(i) + ".bin";
        frames[i] = load_frame(filename.c_str(), width, height);
    }

    for (int i = 0; i < num_frames - 1; ++i) {
        auto& I1 = frames[i];
        auto& I2 = frames[i+1];
        auto flow = horn_schunck(I1, I2, width, height, alpha);

        // Save flow to binary file within the "flow" folder
        FILE* f = fopen(("flow/frame_" + padded(i) + ".bin").c_str(), "wb");
        fwrite(flow.data(), sizeof(float), flow.size(), f);
        fclose(f);
    }

    return 0;
}

