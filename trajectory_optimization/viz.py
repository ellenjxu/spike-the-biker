import matlplotlib.pylot as plt
import numpy as np

def make_vid_from_2d_traj(trajectory_array_1, trajectory_array_2=None):
    # Initialize a list to store images
    images = []

    # Set up the figure and the axis
    fig, ax = plt.subplots()
    plt.axis("off")

    for t in range(trajectory_array_1.shape[0]):
        ax.clear()
        ax.plot(trajectory_array_1[t, :, 1], trajectory_array_1[t, :, 0], "bo")

        if trajectory_array_2 is not None:
            ax.plot(trajectory_array_2[t, :, 1], trajectory_array_2[t, :, 0], "go")

        min_x = min(
            np.min(trajectory_array_1[:, :, 1]),
            np.min(trajectory_array_2[:, :, 1] if trajectory_array_2 is not None else np.inf),
        )
        max_x = max(
            np.max(trajectory_array_1[:, :, 1]),
            np.max(trajectory_array_2[:, :, 1] if trajectory_array_2 is not None else -np.inf),
        )

        min_y = min(
            np.min(trajectory_array_1[:, :, 0]),
            np.min(trajectory_array_2[:, :, 0] if trajectory_array_2 is not None else np.inf),
        )
        max_y = max(
            np.max(trajectory_array_1[:, :, 0]),
            np.max(trajectory_array_2[:, :, 0] if trajectory_array_2 is not None else -np.inf),
        )

        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(data)

    return np.array(images)