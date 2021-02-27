import open3d as o3d
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt

def update_view(camera, intrinsics, extrinsic, width, height):
    z_near = camera.get_near()
    z_far = camera.get_far()

    camera.set_projection(intrinsics, z_near, z_far, width, height)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    eye = - R.T @ t
    up = -extrinsic[1, :3]     # Y camera axis in world frame
    front = -extrinsic[2, :3]  # Z camera axis in world frame
    center = eye - front       # any point on the ray through the camera center

    camera.look_at(center, eye, up)

def main():
    print(o3d.__version__)

    # options
    plot_result = True
    render_depth = True
    if render_depth:
        SHADER = 'depth'
        SUN_LIGHT = False
        FILENAME = "capture_depth_image_filament.png"
    else:
        SHADER = 'defaultUnlit'
        SUN_LIGHT = True
        FILENAME = "capture_screen_image_filament.png"

    # load pointcloud
    pcd = o3d.io.read_point_cloud("fragment.ply")

    # set up viewer
    WIDTH = 1024
    HEIGHT = 768
    render = rendering.OffscreenRenderer(WIDTH, HEIGHT)
    mat = rendering.Material()
    mat.shader = SHADER
    render.scene.add_geometry("__model__", pcd, mat)
    render.scene.scene.enable_sun_light(SUN_LIGHT)
    render.scene.show_axes(False)

    # load pose
    param = o3d.io.read_pinhole_camera_parameters("view_point.json")
    intrinsic = param.intrinsic.intrinsic_matrix
    extrinsic = param.extrinsic
    update_view(render.scene.camera, intrinsic, extrinsic, WIDTH, HEIGHT)

    # render image
    if render_depth:
        # pixels range from 0 (near plane) to 1 (far plane)
        img = render.render_to_depth_image()

        # conversion: [0,1] -> [near_plane, far_plane]
        z_near = render.scene.camera.get_near()
        z_far = render.scene.camera.get_far()

        # img_numpy = (z_far - z_near) * np.asarray(img) + z_near

        # # https://stackoverflow.com/a/62792952
        # numerator = 2.0 * z_near * z_far
        # denominator = z_far + z_near - (2.0 * np.asarray(img) - 1.0) * (z_far - z_near)
        # img_numpy = numerator / denominator

        # conversion inside Open3D:
        # https://github.com/intel-isl/Open3D/blob/b467935323ca3f9b0d9b2672438bb88e396fd325/cpp/open3d/visualization/visualizer/VisualizerRender.cpp#L410-L413
        img_numpy = 2.0 * z_near * z_far / (z_far + z_near - (2.0 * np.asarray(img) - 1.0) * (z_far - z_near))
        img = o3d.geometry.Image(img_numpy)

    else:
        img = render.render_to_image()

    # o3d.io.write_image(FILENAME, img)

    # plot image
    if plot_result:
        plt.imshow(img)
        plt.show()


    #
    # Old Open3D visualizer
    #

    # create window
    vis = VisOpen3D(width=WIDTH, height=HEIGHT, visible=False)

    # point cloud
    vis.add_geometry(pcd)

    # save view point to file
    # vis.save_view_point("view_point.json")
    vis.load_view_point("view_point.json")

    # capture images
    depth = vis.capture_depth_float_buffer(show=False)

    depth_np = np.asarray(depth)
    img_np = np.asarray(img)
    depth_np[400,400]   # 3.110481
    img_np[400,400]     # 0.010031219

    diff = np.asarray(img) - np.asarray(depth)
    # a = diff
    # ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
    # depth_np[ind]

    plt.imshow(diff)
    plt.show()


if __name__ == "__main__":
    main()
