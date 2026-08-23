import argparse
import json
import numpy as np

from nuscenes.nuscenes import NuScenes
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

DEFAULT_PROJECT_ROOT = Path(
    "/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project"
)
CAMERAS = [
	"CAM_FRONT_LEFT",
	"CAM_FRONT",
	"CAM_FRONT_RIGHT",
	"CAM_BACK_LEFT",
	"CAM_BACK",
	"CAM_BACK_RIGHT",
	]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--nuscenes_root", type=Path, default=f"{DEFAULT_PROJECT_ROOT}/datasets/nuscenes-mini")
    parser.add_argument("--detection_path_3D", 
                        type=Path, 
                        default="./detections/transfusion_opencpdet_nuscenes-mini_simpletrack_format_val.json")
    parser.add_argument("--detection_path_2D", 
                        type=Path, 
                        default="./detections/predictions.json")
    return parser.parse_args()

def load_detections(path):
    with open(path, "r") as f:
         data = json.load(f)
    return data

def transform_3D_to_2D(detection, lidar_to_global, cam_pose, cam_cs):
    x, y, z, yaw, length, width, height, score = detection["bbox_3d"]
    box = Box(center=[x, y, z], 
	     size=[width, length, height], 
	     orientation=Quaternion(axis=[0, 0, 1], radians=yaw), 
	     score=score, 
	     name=detection["label"])
    T = np.asarray(lidar_to_global)
    R_lidar_global = T[:3, :3]
    t_lidar_global = T[:3, 3]
    box.rotate(Quaternion(matrix=R_lidar_global))
    box.translate(t_lidar_global)
    box.translate(
	     -np.asarray(cam_pose["translation"])
	     )
    box.rotate(
	     Quaternion(cam_pose["rotation"]).inverse
	     )
    box.translate(
	     -np.asarray(cam_cs["translation"])
	     )
    box.rotate(
	     Quaternion(cam_cs["rotation"]).inverse
	     )
    return box

def compute_center_based_distance():
    pass

def main():
    args = parse_args()
    detections_3D = load_detections(path=args.detection_path_3D)
    nusc = NuScenes(version=str(args.version), dataroot=args.nuscenes_root, verbose=False)
    for detection_3D in detections_3D["frames"]:
        for camera_pose in CAMERAS: 
            sample_token = detection_3D["sample_token"]
            
            cam_token = nusc.get("sample", sample_token)["data"][camera_pose]
            cam_sd = nusc.get("sample_data", cam_token)
            cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"]) 
            cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
            
            detection_3D_in_2D = [transform_3D_to_2D(detection=detection, lidar_to_global=detection_3D["lidar_to_global"], cam_pose=cam_pose, cam_cs=cam_cs) for detection in detection_3D["detections"]]
#             detections_2D = next(
#                                 (detection 
#                                  for detection in load_detections(path=args.detection_path_2D)
#                                  if nusc.get("sample", nusc.get("sample_data", detection["image_id"])["sample_token"])["data"]["LIDAR_TOP"] == sample_token), None)
            detections_2D = load_detections(path=args.detection_path_2D)
            for detection in detections_2D:
                if nusc.get("sample", nusc.get("sample_data", detection["image_id"])["sample_token"])["data"]["LIDAR_TOP"] == nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]:
                    breakpoint()
                    break 

            for detection_2D in detections_2D:
                center_based_distance = compute_center_based_distance(detection_3D_in_2D, detection_2D)
                if center_based_distance > 1.5:
                   continue 
                if detection_3D_in_2D["confidence"] < 0.2:
                   detection_3D_in_2D["confidence"] += detection_2D["confidence"]

    with open(OUTPUT_FILE, "w") as file:
        io.write(maxxed_detections, file)

if __name__ == "__main__":
    main()
