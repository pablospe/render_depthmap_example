import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

import depthmap
from visualization import Visualizer


def main():
    w = 1024
    h = 768

    intrinsic = np.array([[665.10751011,   0.        , 511.5],
                          [  0.        , 665.10751011, 383.5],
                          [  0.        ,   0.        ,   1. ]])

    extrinsic = np.array([[ 0.95038793,  0.0954125 , -0.29607301, -1.84295291],
                          [-0.1222884 ,  0.98976322, -0.07358197, -1.2214318 ],
                          [ 0.28602154,  0.10613772,  0.95232687,  0.6428006 ],
                          [ 0.        ,  0.        ,  0.        ,  1.        ]])


    v = Visualizer(w,h, offscreen=True)
    # v = Visualizer(w,h, off_screen=False)
    mesh = pv.read("fragment_mesh.ply")
    v.add_geometry(mesh)

    # update view
    v.update_view_point(intrinsic, extrinsic)

    # create a camera
    # v.draw_camera(intrinsic, extrinsic)

    # capture screen
    # img = v.capture_screen_image()

    # get depth map image and save to .png
    filename="test.png"
    max_depth_value = 1000
    img = v.capture_depth_image(filename,
                                max_depth_value=max_depth_value)

    # read depth map image from .png
    img2 = depthmap.read_compressed(filename,
                                    max_depth_value=max_depth_value)

    # %matplotlib qt
    # %matplotlib inline
    plt.figure()
    # plt.imshow(img)
    plt.imshow(img2)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

    print("diff:", np.linalg.norm(img - img2))


    #
    # ray casting (testing)
    #
    depth = img
    x = 530
    y = 435
    z = depth[y,x]

    # o = v.plotter.mesh.obbTree
    point3D_world, depth_value = \
        v.ray_cast(x, y, intrinsic, extrinsic,
                   plot=True, max_intersection_distance=5)

    print("Value from depth map:", z)
    print("Value from ray cast: ", depth_value)
    print("Diff:                ", depth_value - z)
    # Note: it seems ray casing is less precise than computing the depth map


if __name__ == "__main__":
    main()
