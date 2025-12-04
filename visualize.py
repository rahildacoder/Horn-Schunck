import numpy as np
import cv2

FLOW_FILE = "flow.bin"
WIDTH = 1920
HEIGHT = 1080
STEP = 20
SCALE = 5


def load_flow(filename, width, height):
    flow = np.fromfile(filename, dtype=np.float32)

    expected = width * height * 2
    if flow.size != expected:
        raise ValueError(
            f"Flow file has {flow.size} floats, expected {expected}. "
            f"Check WIDTH/HEIGHT."
        )

    return flow.reshape((height, width, 2))


def visualize_heatmap(flow):
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    mag = np.sqrt(u * u + v * v)

    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_norm = np.uint8(mag_norm)

    return cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)


def visualize_arrows(flow, step=20, scale=5):
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    h, w = u.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx = u[y, x]
            dy = v[y, x]

            end_x = int(x + dx * scale)
            end_y = int(y + dy * scale)

            cv2.arrowedLine(
                canvas,
                (x, y),
                (end_x, end_y),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

    return canvas


def main():
    print("Loading flow.bin...")
    flow = load_flow(FLOW_FILE, WIDTH, HEIGHT)

    print("Generating heatmap...")
    heatmap = visualize_heatmap(flow)

    print("Generating arrow visualization...")
    arrows = visualize_arrows(flow, STEP, SCALE)

    print("Displaying results...")
    cv2.imshow("Optical Flow Heatmap", heatmap)
    cv2.imshow("Optical Flow Arrows", arrows)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
