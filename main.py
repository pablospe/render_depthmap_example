import open3d
from visualization import VisOpen3D


def main():
    w = 1024
    h = 768

    pcd = open3d.io.read_point_cloud("fragment.ply")

    # create window
    window_visible = True
    vis = VisOpen3D(width=w, height=h, visible=window_visible)

    # point cloud
    vis.add_geometry(pcd)

    # update view
    # vis.update_view_point(intrinsic, extrinsic)

    # save view point to file
    # vis.save_view_point("view_point.json")
    vis.load_view_point("view_point.json")


    # capture images
    depth = vis.capture_depth_float_buffer(show=True)
    image = vis.capture_screen_float_buffer(show=True)

    # save to file
    vis.capture_screen_image("capture_screen_image.png")
    vis.capture_depth_image("capture_depth_image.png")


    # draw camera
    if window_visible:
        vis.load_view_point("view_point.json")
        intrinsic = vis.get_view_point_intrinsics()
        extrinsic = vis.get_view_point_extrinsics()
        vis.draw_camera(intrinsic, extrinsic, scale=0.5, color=[0.8, 0.2, 0.8])
        # vis.update_view_point(intrinsic, extrinsic)

    if window_visible:
        vis.load_view_point("view_point.json")
        vis.run()

    del vis


if __name__ == "__main__":
    main()
