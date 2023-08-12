import cv2
import numpy as np
import sys
import argparse
import glob
import os

GRID_ROWS = 5
GRID_COLS = 8

HEIGHT = 512 // 2
WIDTH = 1024 // 2

OUTPUT_FN = "selected.png"


def run_picker(folder):
    img_fns = glob.glob(os.path.join(folder, "*.png"))
    images = [cv2.imread(img_fn) for img_fn in img_fns]

    grid_size = (HEIGHT * GRID_ROWS, WIDTH * GRID_COLS, 3)
    grid = np.zeros(grid_size, dtype=np.uint8)

    def reset_grid():
        for i, img in enumerate(images):
            row = i // GRID_COLS
            col = i % GRID_COLS
            grid[row * HEIGHT : (row + 1) * HEIGHT, col * WIDTH : (col + 1) * WIDTH] = cv2.resize(img, (WIDTH, HEIGHT))

    reset_grid()

    cv2.namedWindow(folder)

    complete = [False]

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            col = x // WIDTH
            row = y // HEIGHT
            index = row * GRID_COLS + col
            print(f"You clicked on image {index} {img_fns[index]}")
            cv2.imwrite(os.path.join(folder, OUTPUT_FN), images[index])
            complete[0] = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            col = x // WIDTH
            row = y // HEIGHT
            index = row * GRID_COLS + col
            img = images[index]
            grid[0 : img.shape[0], 0 : img.shape[1]] = img

    cv2.setMouseCallback(folder, click_event)

    while not complete[0]:
        cv2.imshow(folder, grid)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            sys.exit(0)
        elif key == ord("r"):
            reset_grid()
        elif key == ord("s"):
            break

    cv2.destroyAllWindows()


def main(data_path):
    for folder in glob.iglob(os.path.join(data_path, "*")):
        if os.path.exists(os.path.join(folder, OUTPUT_FN)):
            continue
        run_picker(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    main(args.data_path)
