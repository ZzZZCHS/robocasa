import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
import glm
import os
from tqdm import tqdm
import glob
import json


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
    


data_dir = "/ailab/user/huanghaifeng/work/robocasa_exps_haifeng/robocasa/robocasa/models/assets/objects/aigen_objs"
failed_objs = []
for obj_class in tqdm(os.listdir(data_dir)):
    print('processing', obj_class)
    obj_class_dir = os.path.join(data_dir, obj_class)
    for obj_name in tqdm(os.listdir(obj_class_dir)):
        if obj_name.startswith('._'):
            os.remove(os.path.join(obj_class_dir, obj_name))
            continue
        if not obj_name.startswith(obj_class):
            continue
        obj_dir = os.path.join(obj_class_dir, obj_name)
        save_path = os.path.join(obj_dir, 'combined_views.png')
        obj_files = glob.glob(os.path.join(obj_dir, 'visual/*.obj'))
        try:
            process_obj(obj_files, save_path)
        except KeyboardInterrupt:
            exit(0)
        except Exception as e:
            print('fail', obj_dir)
            failed_objs.append(f"{obj_class}/{obj_name}")

with open("failed_to_render.json", 'w') as f:
    json.dump(failed_objs, f, indent=4)
        
# PYOPENGL_PLATFORM=egl python robocasa/scripts/render_objects.py