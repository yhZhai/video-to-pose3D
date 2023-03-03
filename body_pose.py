"""
A Class for Human Body Pose Definition & Visualization
"""

from matplotlib import pyplot as plt
import numpy as np


class HumanPose(object):
    def __init__(self):
        """
        Initialization for the basic joint and connection information
        """

        # Basic Joint & Connection Definition
        # based on Human3.6M (http://vision.imar.ro/human3.6m/description.php)
        self.total_joint_num = 17
        self.joints_name = (
            "Pelvis",
            "R_Hip",
            "R_Knee",
            "R_Ankle",
            "L_Hip",
            "L_Knee",
            "L_Ankle",
            "Torso",
            "Neck",
            "Nose",
            "Head_top",
            "L_Shoulder",
            "L_Elbow",
            "L_Wrist",
            "R_Shoulder",
            "R_Elbow",
            "R_Wrist",
        )
        # Paired Joints
        self.flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        # Root Joint
        self.root_joint_idx = self.joints_name.index("Pelvis")

        # Bone Connection/Edge & its Color
        self.skeleton = [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 4),
            (4, 5),
            (5, 6),
            (0, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (8, 11),
            (11, 12),
            (12, 13),
            (8, 14),
            (14, 15),
            (15, 16),
        ]
        self.color = [
            (0 / 255.0, 134 / 255.0, 255 / 255.0),
            (0 / 255.0, 215 / 255.0, 255 / 255.0),
            (0 / 255.0, 255 / 255.0, 204 / 255.0),
            (0 / 255.0, 134 / 255.0, 255 / 255.0),
            (0 / 255.0, 215 / 255.0, 255 / 255.0),
            (77 / 255.0, 255 / 255.0, 222 / 255.0),
            (255 / 255.0, 127 / 255.0, 77 / 255.0),
            (107 / 255.0, 222 / 255.0, 69 / 255.0),
            (83 / 255.0, 128 / 255.0, 227 / 255.0),
            (199 / 255.0, 207 / 255.0, 50 / 255.0),
            (94 / 255.0, 102 / 255.0, 102 / 255.0),
            (219 / 255.0, 83 / 255.0, 185 / 255.0),
            (93 / 255.0, 41 / 255.0, 227 / 255.0),
            (94 / 255.0, 102 / 255.0, 102 / 255.0),
            (219 / 255.0, 83 / 255.0, 185 / 255.0),
            (93 / 255.0, 41 / 255.0, 227 / 255.0),
        ]

    def drawer_3d(self, ax, viz_3d):
        """
        Visualiation for 3D body pose
        Args:
            ax:     axes/subplot
            viz_3d: [3, 17] 3D array of body pose
        """
        ax.scatter(viz_3d[0], viz_3d[2], viz_3d[1], "b.")
        for idx, edge in enumerate(self.skeleton):
            ax.plot(
                np.concatenate(
                    [viz_3d[0][edge[0]].reshape(1), viz_3d[0][edge[1]].reshape(1)]
                ),
                np.concatenate(
                    [viz_3d[2][edge[0]].reshape(1), viz_3d[2][edge[1]].reshape(1)]
                ),
                np.concatenate(
                    [viz_3d[1][edge[0]].reshape(1), viz_3d[1][edge[1]].reshape(1)]
                ),
                color=self.color[idx][::-1],
            )

    def drawer_2d(self, ax, viz_2d):
        """
        Visualiation for 2D body pose
        Args:
            ax:     axes/subplot
            viz_2d: [2, 17] 2D array of body pose
        """
        ax.plot(viz_2d[0], viz_2d[1], "b.")
        for idx, edge in enumerate(self.skeleton):
            ax.plot(
                np.concatenate(
                    [viz_2d[0][edge[0]].reshape(1), viz_2d[0][edge[1]].reshape(1)]
                ),
                np.concatenate(
                    [viz_2d[1][edge[0]].reshape(1), viz_2d[1][edge[1]].reshape(1)]
                ),
                color=self.color[idx][::-1],
            )

    def compute_mpjpe(self, body_joint_1, body_joint_2):
        """
        Evaluate mean per-joint position error (mpjpe) between 2 poses
        Args:
            body_joint_1: [JOINT_NUM, JOINT_DIM]
            body_joint_2: [JOINT_NUM, JOINT_DIM]
        Return:
            pose_err: pose error
        """
        # error metric
        pose_err = np.sqrt(np.sum((body_joint_1 - body_joint_2) ** 2, axis=1)).mean()
        return pose_err


if __name__ == "__main__":
    # Example 3D human pose as a 2D array with shape (17, 3)
    # Each 3D vector is the 3D coordinate (x, y, z) of each joint
    test_3d_pose = [
        [304.5, 410.5, 200.0],
        [252.0, 407.0, 200.0],
        [255.0, 586.0, 200.0],
        [279.0, 745.0, 200.0],
        [357.0, 414.0, 200.0],
        [385.0, 603.0, 200.0],
        [452.0, 766.0, 200.0],
        [310.25, 290.25, 200.0],
        [316.0, 170.0, 200.0],
        [326.0, 105.0, 200.0],
        [296.0, 85.0, 200.0],
        [390.0, 169.0, 200.0],
        [407.0, 295.0, 200.0],
        [448.0, 402.0, 200.0],
        [243.0, 168.0, 200.0],
        [174.0, 264.0, 200.0],
        [213.0, 310.0, 200.0],
    ]
    test_3d_pose = np.array(test_3d_pose)

    # Create a instance of HumanPose Class
    pose_instance = HumanPose()
    # Create a new figure
    f = plt.figure()
    # Add an axes/subplot to the figure for 3D pose visualization
    ax = f.add_subplot(131, projection="3d")
    plt.gca().invert_zaxis()
    pose_instance.drawer_3d(ax, test_3d_pose.transpose())
    ax.title.set_text("test_3d_pose")

    # Add an axes/subplot to the figure for 2D pose visualization
    ax = f.add_subplot(133)
    plt.gca().invert_yaxis()
    pose_instance.drawer_2d(ax, test_3d_pose[:, :2].transpose())
    ax.title.set_text("test_2d_pose")

    plt.tight_layout()
    plt.show

    # calculate error between two given 3D body poses
    err_3d_pose = pose_instance.compute_mpjpe(test_3d_pose, test_3d_pose)
    print(f"The mpjpe for the input two 3D body poses is {err_3d_pose}")

    # calculate error between two given 2D body poses
    err_2d_pose = pose_instance.compute_mpjpe(test_3d_pose[:, :2], test_3d_pose[:, :2])
    print(f"The mpjpe for the input two 2D body poses is {err_2d_pose}")
