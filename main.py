import open3d as o3d
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt

def update_view(camera, intrinsics, extrinsic, width, height):
    camera.set_projection(intrinsics, 0.1, 100, width, height)
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
    render_depth = False
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
    img = render.render_to_image()
    img = render.render_to_depth_image()
    o3d.io.write_image(FILENAME, img, 9)

    # plot image
    if plot_result:
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    main()
