from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from spike_the_biker.camera import distort_image


def split_list(input_list, sublist_length):
    return [input_list[i : i + sublist_length] for i in range(0, len(input_list), sublist_length)]


if __name__ == "__main__":
    INPUT_DIR = Path("../../data/render_100k/images")
    OUTPUT_DIR = Path("../../data/render_100k/processed_images")
    EXT = ".jpg"
    THREADS = 6

    OUTPUT_DIR.mkdir(exist_ok=True)

    # comma_hack_4
    x, y, w, h = 103, 60, 270, 167

    K = np.array([[ 5.97615131e+02, -4.50346295e-01,  9.43058151e+02],
                      [ 0.00000000e+00,  5.97320118e+02,  5.66164841e+02],
                      [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    D = np.array([-0.03800249,
                    0.04716124,
                    -0.04492046,
                    0.01374281])

    scaled_K = K / 4
    scaled_K[2][2] = 1

    def process_paths(paths, job_index):
        for i, pth in enumerate(paths):
            # TODO not the best to report progress, can we do smth better?
            if i % 500 == 0:
                print(f"job {job_index}: {i}/{len(paths)}")
            if pth.suffix == EXT:
                out_pth = OUTPUT_DIR / pth.name

                img = cv2.imread(str(pth))
                dist = distort_image(img, scaled_K, D, crop_output=False)
                crop_img = dist[y : y + h, x : x + w]
                cv2.imwrite(str(out_pth), crop_img)

    # split into multiple jobs
    all_paths = [pth for pth in INPUT_DIR.iterdir()]
    job_paths = split_list(all_paths, sublist_length=(len(all_paths) // THREADS))

    Parallel(n_jobs=THREADS)(
        delayed(process_paths)(job_paths[i], i) for i in range(len(job_paths))
    )
