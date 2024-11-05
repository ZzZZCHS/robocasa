import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
import glm
import os
from tqdm import tqdm
import imageio
import glob
import json
import logging
from datetime import datetime
from termcolor import cprint
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2

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

z_height = 0.2

views = {
    'view_front': np.linalg.inv(np.array(glm.lookAt(
        glm.vec3(0.0, z_height, z_height),
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(0.0, 0.0, 1)
    ))),
    'view_back': np.linalg.inv(np.array(glm.lookAt(
        glm.vec3(0.0, -z_height, z_height),
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(0.0, 0.0, 1.0)
    ))),
    'view_left': np.linalg.inv(np.array(glm.lookAt(
        glm.vec3(z_height, 0.0, z_height),
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(0.0, 0.0, 1.0)
    ))),
    'view_right': np.linalg.inv(np.array(glm.lookAt(
        glm.vec3(-z_height, 0.0, z_height),
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(0.0, 0.0, 1.0)
    ))),
}

def render_view(scene, renderer, camera_pose):
    scene.set_pose(scene.main_camera_node, pose=camera_pose)
    color, _ = renderer.render(scene)
    return color.copy()


def process_obj(obj_files, video_writer, object_name):
    print(obj_files)
    print(save_path)
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
        [0.0, 0.0, 1.0, 0.0],
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

    fig.text(0.05, 0.95, object_name, ha='left', va='top', fontsize=16, color='white', weight='bold',
         bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    renderer.delete()


def process_extra(obj_files, video_writer, object_name):
    outputs = dict()
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
        [0.0, 0.0, 1.0, 0.0],
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

    fig.text(0.05, 0.95, object_name, ha='left', va='top', fontsize=16, color='white', weight='bold',
         bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

    # Render to an in-memory buffer and write to video
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Convert buffer to an image for video_writer
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_writer.append_data(frame)

    plt.close(fig)
    renderer.delete()
    
path="/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/robocasa/models/assets/objects/objaverse_extra"
type_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
type_dirs = sorted(type_dirs)
save_dir="/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/robocasa/models/assets/objects/filtered_img"

count = 0
video_path="/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/robocasa/models/assets/objects/filtered_img/extra.mp4"
video_writer=imageio.get_writer(video_path, fps=2)

for type in type_dirs:
    count += 1

    type_dir = os.path.join(path, type)
    objs = [d for d in sorted(os.listdir(type_dir)) if os.path.isdir(os.path.join(type_dir, d))]
    for obj in objs:
        obj_dir = os.path.join(type_dir, obj)
        # save_path = os.path.join(save_dir, type, obj)
        # os.makedirs(save_path, exist_ok=True)
        # save_path = os.path.join(save_path, 'combined_views.png')
        obj_files = glob.glob(os.path.join(obj_dir, 'visual/*.obj'))
        logging.info(f"[RENDER] render objects from {obj_dir}")
        try:
            # process_obj(obj_files, save_path, obj)
            process_extra(obj_files, video_writer, obj)
            logging.info(f"[SUCCESS] success to render objects from {obj_dir}")
        except Exception as e:
            print(e)
            logging.error(e)
            logging.error(f"[FAIL] fail to render objects from {obj_dir}")

video_writer.close()
print(f"TOTAL CATEGORIES: {count}")

# obj_dir="/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/robocasa/models/assets/objects/objaverse_extra/tissue_box/tissue_box_0"
# save_path = "/ssd/home/groups/smartbot/huanghaifeng/robocasa_exps/robocasa/robocasa/scripts/test.png"
# obj_files = glob.glob(os.path.join(obj_dir, 'visual/*.obj'))
# logging.info(f"[RENDER] render objects from {obj_dir}")
# try:
#     process_obj(obj_files, save_path)
#     logging.info(f"[SUCCESS] success to render objects from {obj_dir}")
# except Exception as e:
#     print(e)
#     logging.error(e)
#     logging.error(f"[FAIL] fail to render objects from {obj_dir}")


# PYOPENGL_PLATFORM=egl python robocasa/scripts/render_objects_2.py
# PYOPENGL_PLATFORM=egl python render_objects_2.py