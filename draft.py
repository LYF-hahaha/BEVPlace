import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def test():
    path = "/home/alex/Dataset/Apollo/SanJoseDowntown_TrainData/pcds/1.pcd"

    vis = o3d.visualization.Visualizer()
    vis.create_window("XXXYYYZZZ", 1024, 768)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    opt.show_coordinate_frame = False

    pcd = o3d.io.read_point_cloud(path)

    vis.clear_geometries()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(axis)
    vis.add_geometry(pcd)

    # gui.Application.instance.create_window('My First Window', 800, 600)
    # gui.Application.instance.run()
    vis.run()
    # time.sleep(10)


class App:
    def __init__(self):
        # 初始化实例
        gui.Application.instance.initialize()

        # 创建主窗口
        self.window = gui.Application.instance.create_window('My First Window', 800, 600)

        # 创建显示场景
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        # 将场景添加到窗口中
        self.window.add_child(self.scene)

        # 创建一个球
        sphere = o3d.geometry.TriangleMesh.create_sphere()
        sphere.paint_uniform_color([0.0, 1.0, 1.0])
        sphere.compute_vertex_normals()
        material = rendering.MaterialRecord()
        material.shader = 'defaultLit'

        # 将球加入场景中渲染
        self.scene.scene.add_geometry("Sphere", sphere, material)

        # 设置相机属性
        bounds = sphere.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())

    def run(self):
        gui.Application.instance.run()


def make_point_cloud(npts, center, radius):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def high_level():
    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = False
    vis.add_geometry("Points", points)
    for idx in range(0, len(points.points)):
        vis.add_3d_label(points.points[idx], "{}".format(idx))

    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()


if __name__ == "__main__":
    test()

