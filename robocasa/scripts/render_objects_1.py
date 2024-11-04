import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
import glm
import os
from tqdm import tqdm
import glob
import json
import logging
from datetime import datetime
from termcolor import cprint

# 生成带时间戳的日志文件名
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f'./logs/output_{current_time}.log'

# 配置日志设置
logging.basicConfig(
    filename=log_filename,  # 设置日志文件名
    filemode='w',           # 以写模式打开（每次运行会覆盖旧日志）
    level=logging.INFO,     # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 设置日志格式
)


views = {
    'view_front': np.linalg.inv(np.array(glm.lookAt(
        glm.vec3(0.0, 1.0, 1.0),
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(0.0, 0.0, 1.0)
    ))),
    'view_back': np.linalg.inv(np.array(glm.lookAt(
        glm.vec3(0.0, -1.0, 1.0),
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(0.0, 0.0, 1.0)
    ))),
    'view_left': np.linalg.inv(np.array(glm.lookAt(
        glm.vec3(1.0, 0.0, 1.0),
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(0.0, 0.0, 1.0)
    ))),
    'view_right': np.linalg.inv(np.array(glm.lookAt(
        glm.vec3(-1.0, 0.0, 1.0),
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(0.0, 0.0, 1.0)
    ))),
}

def render_view(scene, renderer, camera_pose):
    scene.set_pose(scene.main_camera_node, pose=camera_pose)
    color, _ = renderer.render(scene)
    return color.copy()


def render_image(
        obj_file, 
        save_path,
        transform=None,
        center=True,
        prescale=True,
        rot=None):
    
    cprint(f"render object from {obj_file}")
    # 初始化一个空列表来存储所有加载的子模型
    models = []

    # 遍历 obj_file 列表，逐一加载
    for file in obj_file:
        print(f"Render object from {file}")
        
        resolver = trimesh.resolvers.FilePathResolver(os.path.dirname(file))
        
        # 加载每个 .obj 文件
        obj_model = trimesh.load(
            file,
            resolver=resolver,
            split_object=True,
            process=False,
            maintain_order=False
        )

        # 将加载的模型添加到 models 列表中
        models.append(obj_model)

    # 合并所有子模型为一个整体模型
    model = trimesh.util.concatenate(models)

    rot = rot or []

    if transform is not None:
        assert center is False and prescale is False

    if center:
        mat_t = np.eye((4))
        center = (model.bounds[0] + model.bounds[1]) / 2
        print(f"center: {center}")
        mat_t[:3,3] = -center
        transform = mat_t
    
    if prescale:
        mat_s = np.eye((4))
        scale_factor = 1 / np.max(model.bounds[1] - model.bounds[0])
        mat_s *= scale_factor
        print(f"scale_factor: {scale_factor}")

        if transform is None:
            transform = mat_s
        else:
            transform = np.matmul(mat_s, transform)
    
    for r in rot:
        mat_R = np.eye(4)
        
        if r in ["x", "x90"]:
            mat_R[:3,:3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        elif r in ["x180"]:
            mat_R[:3,:3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        elif r in ["x270"]:
            mat_R[:3,:3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        
        elif r in ["y", "y90"]:
            mat_R[:3,:3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        elif r in ["y180"]:
            mat_R[:3,:3] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
        elif r in ["y270"]:
            mat_R[:3,:3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        

        elif r in ["z", "z90"]:
            mat_R[:3,:3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        elif r in ["z180"]:
            mat_R[:3,:3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        elif r in ["z270"]:
            mat_R[:3,:3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        elif r in ["z60"]:
            mat_R[:3,:3] = [[0.5000000, -0.8660254,  0.0000000], [0.8660254,  0.5000000,  0.0000000], [0, 0, 1]]
        
        
        else:
            raise ValueError("Invalid choice of rotation {}".format(r))


        if transform is None:
            transform = mat_R
        else:
            transform = np.matmul(mat_R, transform)


    if transform is not None:
        if isinstance(model, trimesh.base.Trimesh):
            model.apply_transform(transform)
        else:
            for k in model.geometry:
                model.geometry[k].apply_transform(transform)

    # mesh_save_path = "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/obj2mjcf/tests/stranger_0/ice_box.obj"
    # model.export(mesh_save_path)
    # return 

    # render the model, save
    combined_scene = trimesh.Scene()
    # combined_scene.add_geometry(model)
    for obj_file in obj_files:
        mesh = trimesh.load(obj_file)
        mesh.apply_transform(transform)
        combined_scene.add_geometry(mesh)


    scene = pyrender.Scene()
    for name, mesh in combined_scene.geometry.items():
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(pyrender_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    for z in [1.]:
        for x in [-2., 2.]:
            for y in [-2., 2.]:
                tmp_light = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0) 
                light_pose = np.linalg.inv(np.array(glm.lookAt(
                    glm.vec3(x, y, 2.0*z),
                    glm.vec3(0.0, 0.0, 0.0),
                    glm.vec3(0.0, 0.0, 1.0)
                )))
                # print(light_pose)
                scene.add(tmp_light, pose=light_pose)

    width = height = 256
    renderer = pyrender.OffscreenRenderer(width, height, point_size=1.0)

    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    # Render and save each view
    for i, (view_name, pose) in enumerate(views.items()):
        color = render_view(scene, renderer, pose)
        axes[i].imshow(color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    renderer.delete()


def process_obj(obj_files, save_path):
    combined_scene = trimesh.Scene()
    for obj_file in obj_files:
        mesh = trimesh.load(obj_file)
        combined_scene.add_geometry(mesh)

    scene = pyrender.Scene()
    for name, mesh in combined_scene.geometry.items():
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(pyrender_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    for z in [1.]:
        for x in [-2., 2.]:
            for y in [-2., 2.]:
                tmp_light = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0) 
                light_pose = np.linalg.inv(np.array(glm.lookAt(
                    glm.vec3(x, y, 2.0*z),
                    glm.vec3(0.0, 0.0, 0.0),
                    glm.vec3(0.0, 0.0, 1.0)
                )))
                # print(light_pose)
                scene.add(tmp_light, pose=light_pose)

    width = height = 256
    renderer = pyrender.OffscreenRenderer(width, height, point_size=1.0)

    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    # Render and save each view
    for i, (view_name, pose) in enumerate(views.items()):
        color = render_view(scene, renderer, pose)
        axes[i].imshow(color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    renderer.delete()
    
path="/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/robocasa/models/assets/objects/objaverse_extra"
type_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
type_dirs = sorted(type_dirs)

for type in type_dirs:
    if type != "wine" and type != "wine_glass":
        continue
    type_dir = os.path.join(path, type)
    objs = [d for d in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, d))]
    for obj in objs:
        obj_dir = os.path.join(type_dir, obj)
        save_path = os.path.join(obj_dir, 'combined_views.png')
        obj_files = glob.glob(os.path.join(obj_dir, 'visual/*.obj'))
        logging.info(f"[RENDER] render objects from {obj_dir}")
        try:
            # process_obj(obj_files, save_path)
            render_image(obj_files, save_path)
            logging.info(f"[SUCCESS] success to render objects from {obj_dir}")
        except Exception as e:
            print(e)
            logging.error(e)
            logging.error(f"[FAIL] fail to render objects from {obj_dir}")

# obj_path="/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/robocasa/models/assets/objects/objaverse/stranger/stranger_0/visual/"
# save_path=os.path.join(obj_path, 'combined_views.png')
# # obj_files = glob.glob(os.path.join(obj_path, 'visual/*.obj'))
# obj_files = glob.glob(os.path.join(obj_path, '*.obj'))
# print(obj_files)
# try:
#     process_obj(obj_files, save_path)
# except Exception as e:
#     print(e)
#     print('fail', obj_path)
    

# PYOPENGL_PLATFORM=egl python robocasa/scripts/render_objects_1.py
# PYOPENGL_PLATFORM=egl python render_objects_1.py