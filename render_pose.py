import os
import time
from pathlib import Path

from common.arguments import parse_args
from common.camera import *
from common.generators import UnchunkedGenerator
from common.loss import *
from common.model import *
from common.utils import Timer, evaluate, add_path

# from joints_detectors.openpose.main import generate_kpts as open_pose


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata = {
    "layout_name": "coco",
    "num_joints": 17,
    "keypoints_symmetry": [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]],
}

add_path()


# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


time0 = ckpt_time()


def get_detector_2d(detector_name):
    def get_alpha_pose():
        from joints_detectors.Alphapose.gene_npz import generate_kpts as alpha_pose

        return alpha_pose

    def get_hr_pose():
        from joints_detectors.hrnet.pose_estimation.video import (
            generate_kpts as hr_pose,
        )

        return hr_pose

    detector_map = {
        "alpha_pose": get_alpha_pose,
        "hr_pose": get_hr_pose,
        # 'open_pose': open_pose
    }

    assert (
        detector_name in detector_map
    ), f"2D detector: {detector_name} not implemented yet!"

    return detector_map[detector_name]()


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]


def main(args):
    '''
    video -> 2D keypoints -> 3D pose -> render
    '''

    zoom_factor = 20
    npy_path = "averages/cluster1_average.npy"
    lift = 1

    dir_name = os.path.dirname(npy_path)
    basename = os.path.basename(npy_path)
    video_name = basename[: basename.rfind(".")]
    args.viz_output = f"{dir_name}/{video_name}.mp4"

    raw_prediction = np.load(npy_path)
    raw_prediction *= zoom_factor
    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    input_prediction = camera_to_world(raw_prediction, R=rot, t=lift)

    # # as we don't have the trajectory, we rebase the height
    anim_output = {"Reconstruction": input_prediction}
    input_keypoints = image_coordinates(raw_prediction[..., :2], w=1000, h=1002)

    from common.visualization import render_animation

    render_animation(
        input_keypoints,
        anim_output,
        Skeleton(),
        25,
        args.viz_bitrate,
        np.array(70.0, dtype=np.float32),
        args.viz_output,
        limit=args.viz_limit,
        downsample=args.viz_downsample,
        size=args.viz_size,
        input_video_path=None,
        viewport=(1000, 1002),
        input_video_skip=args.viz_skip,
        left_title="2D Keypoints",
    )
    print("rendered video saved to ", args.viz_output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
