#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <vector>

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

    // Iterative refinement of flow estimates
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<float> U_avg(width * height, 0.0f);
        std::vector<float> V_avg(width * height, 0.0f);

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

std::vector<float> load_frame(const char* filename, int width, int height) {
    std::vector<float> frame(width * height);
    FILE* file = fopen(filename, "rb");
    fread(frame.data(), sizeof(float), width * height, file);
    fclose(file);
    return frame;
}

int main() {
    const int width = 1920;
    const int height = 1080;

    std::vector<float> I1 = load_frame("frames/frame_0053.bin", width, height);
    std::vector<float> I2 = load_frame("frames/frame_0054.bin", width, height);

    float alpha = 1.0f;

    std::vector<float> flow = horn_schunck(I1, I2, width, height, alpha);

    FILE* f = fopen("flow.bin", "wb");
    fwrite(flow.data(), sizeof(float), flow.size(), f);
    fclose(f);

    return 0;
}

