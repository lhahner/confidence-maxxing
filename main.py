# TODO general problem to think about; if a detection has low confidence and wrong class but bounding box is true 
# what to do?
import argparse
import json
import numpy as np
import math
import torch

from torchvision.ops import nms, box_iou
from data_classes.bounding_box import BoundingBox
from nuscenes.nuscenes import NuScenes
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

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
CATEGORIES = {1: 'barrier', 
              2: 'bicycle', 
              3: 'bus', 
              4: 'car', 
              5: 'construction_vehicle', 
              6: 'motorcycle', 
              7: 'pedestrian', 
              8: 'traffic_cone', 
              9: 'trailer', 
              10: 'truck'}

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
    parser.add_argument("--association_strategy",
                        type=str,
                        default="iou-based")
    return parser.parse_args()

def load_detections(path):
    with open(path, "r") as f:
         data = json.load(f)
    return data

def transform_3D_to_2D(detection, 
		       lidar_to_global, 
                       cam_pose, 
                       cam_cs,
                       imsize=(1600, 900), 
		       min_depth=0.1,
		       sample_token="",
		       camera_token=""):
    x, y, z, yaw, length, width, height, score = detection["bbox_3d"]
    box = Box([x, y, z], [width, length, height],
              Quaternion(axis=[0, 0, 1], radians=yaw),
              score=score, name=detection["label"])

    T = np.asarray(lidar_to_global)
    box.rotate(Quaternion(matrix=T[:3, :3]))
    box.translate(T[:3, 3])

    box.translate(-np.asarray(cam_pose["translation"]))
    box.rotate(Quaternion(cam_pose["rotation"]).inverse)
    box.translate(-np.asarray(cam_cs["translation"]))
    box.rotate(Quaternion(cam_cs["rotation"]).inverse)

    corners = box.corners()
    if np.any(corners[2, :] <= min_depth):
        return None

    K = np.asarray(cam_cs["camera_intrinsic"], dtype=float)
    pts = view_points(corners, K, normalize=True)[:2, :]

    cx, cy = view_points(box.center.reshape(3, 1), K, normalize=True)[:2, 0]

    rx1, ry1 = pts[0].min(), pts[1].min()
    rx2, ry2 = pts[0].max(), pts[1].max()

    W, H = imsize
    if rx2 <= 0 or rx1 >= W or ry2 <= 0 or ry1 >= H:
        return None
    x1, x2 = np.clip([rx1, rx2], 0, W - 1)
    y1, y2 = np.clip([ry1, ry2], 0, H - 1)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return BoundingBox(
		corners_2D=[float(x1), float(y1), float(x2), float(y2)],
    	 	corners_3D=corners,
		centers_2D=[float(cx), float(cy)],
		size_3D=[width, length, height],
		size_2D=[x2 - x1, y2 - y1],
		label=detection["label"],
		confidence_score=float(score),
		nuscenes_camera_token=camera_token,
		nuscenes_sample_token=sample_token)	

def is_bounding_box_in_camera_frame(detection, nusc, camera_token, intrinsic, cam_sd):
    x, y, z, yaw, length, width, height, score = detection["bbox_3d"]
    box = Box(
        center=[x, y, z],
        size=[width, length, height],
        orientation=Quaternion(axis=[0, 0, 1], radians=yaw)
    )
    box = box_global_to_camera(
            nusc,
            box,
            camera_token
    )
    visible = box_in_image(
            box,
            intrinsic,
            (cam_sd["width"], cam_sd["height"]),
            vis_level=BoxVisibility.ANY
    )
    return visible

def detection_in_camera(nusc, frame, detection, camera_name):
    
    #if not "bbox_3d" in detection:
    bbox = detection["bbox_3d"]

    x, y, z = bbox[0:3]
    yaw = bbox[3]
    length = bbox[4]
    width = bbox[5]
    height = bbox[6]

    center = np.array([
    	x,
        y,
        z + height / 2
    ])
    size = np.array([
        width,
        length,
        height
    ])
    orientation = Quaternion(
        axis=[0, 0, 1],
        radians=yaw
    )
    box = Box(
        center=center,
        size=size,
        orientation=orientation
    )
    lidar_to_global = np.asarray(frame["lidar_to_global"])
    R_lidar_global = lidar_to_global[:3, :3]
    t_lidar_global = lidar_to_global[:3, 3]

    box.rotate(Quaternion(matrix=R_lidar_global))
    box.translate(t_lidar_global)

    sample = nusc.get(
        "sample",
        frame["sample_token"]
    )

    cam_token = sample["data"][camera_name]
    cam_data = nusc.get(
        "sample_data",
        cam_token
    )
    ego_pose = nusc.get(
        "ego_pose",
        cam_data["ego_pose_token"]
    )
    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        cam_data["calibrated_sensor_token"]
    )
    box.translate(
        -np.asarray(ego_pose["translation"])
    )
    box.rotate(
        Quaternion(ego_pose["rotation"]).inverse
    )
    box.translate(
        -np.asarray(calibrated_sensor["translation"])
    )
    box.rotate(
        Quaternion(calibrated_sensor["rotation"]).inverse
    )
    camera_intrinsic = np.asarray(
        calibrated_sensor["camera_intrinsic"]
    )
    return box_in_image(
        box,
        camera_intrinsic,
        (cam_data["width"], cam_data["height"]),
        vis_level=BoxVisibility.ANY
    )

def box_global_to_camera(nusc, box, camera_token):
    cam_sd = nusc.get("sample_data", camera_token)

    ego_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        cam_sd["calibrated_sensor_token"]
    )

    box.translate(-np.array(ego_pose["translation"]))
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)

    box.translate(-np.array(calibrated_sensor["translation"]))
    box.rotate(Quaternion(calibrated_sensor["rotation"]).inverse)

    return box

def compute_iou_based_distance(transformed_2D_bbox, native_2D_bbox):
    return box_iou(torch.tensor([native_2D_bbox]), torch.tensor([transformed_2D_bbox])).item() # IoU can also take multiple boxes

def compute_center_based_distance(box_a, box_b):
    matching_boxes = []

    for a_box in box_a:
        if a_box is None:
           for b_box in box_b:
               distance_norm = math.dist(a_box.center, b_box.center) / math.sqrt(a_box.wlh[0]**2 + a_box.wlh[2]**2)
               if distance_norm < 0.25:
                  matching_boxes.append(a_box)
    return matching_boxes

# TODO Here we should implement a module which is exchangable so that various optimization alogirhtns can be used.
def associate_detections(detection_3D_in_2D, detections_2D, threshold):
    matches = []
    for detection_2D in detections_2D:
        # TODO reconstruct as strategy pattern, different distance strategies
        #if detection_3D_in_2D.confidence_score == 0.05057337135076523 and detection_2D.confidence_score == 0.5404:
         #   breakpoint()
        intersection = compute_iou_based_distance(detection_3D_in_2D.corners_2D, detection_2D.corners_2D)
        if intersection >= threshold and detection_2D.label == detection_3D_in_2D.label: 
            pair = (detection_2D, detection_3D_in_2D)
            matches.append(pair)
    #if detection_3D_in_2D.confidence_score == 0.05057337135076523:
    #    breakpoint()
    return matches

def apply_nms(matches):
    """
    Expects a 1x2 list which contains matching bounding boxes. Since the second
    Bounding-Box is always the same we compute NMS only with the (N,1) and (1,2)
    >>> E.g.:
        [[detection_2D: BoundingBox(...), detection_3D_in_2D: BoundingBox(...)],
         [detection_2D: BoundingBox(...), detection_3D_in_2D: BoundingBox(...)]]
    
    Returns
    ---
        The best matching tuple of matches computed by NMS.
    """
    detections_2D, detections_3D_in_2D = zip(*matches) # Unpack the list of tuples

    corners_2D_tensor = torch.tensor([detection.corners_2D for detection in detections_2D])
    scores_tensor = torch.tensor([detection.confidence_score for detection in detections_2D])

    boxes = torch.concat((corners_2D_tensor, torch.tensor([detections_3D_in_2D[0].corners_2D])))
    scores = torch.concat((scores_tensor, torch.tensor([detections_3D_in_2D[0].confidence_score])))
    
    keep_indices = nms(boxes, scores, iou_threshold=0.8)
    return matches[keep_indices[0]]

def maximise_confidence(detection_path_3D, 
                        detection_path_2D,
                        version="v1.0-mini",
                        nuscenes_root=f"{DEFAULT_PROJECT_ROOT}/datasets/nuscenes-mini",
                        association_strategy="iou-based"): # Currently not really used
    frames_3D = load_detections(path=detection_path_3D)
    nusc = NuScenes(version=str(version), dataroot=nuscenes_root, verbose=False)
    frames_3D_copy = frames_3D
    for frame_3D in frames_3D["frames"]:
        for camera_pose in CAMERAS: 
            sample_token = frame_3D["sample_token"]
            
            cam_token = nusc.get("sample", sample_token)["data"][camera_pose]
            cam_sd = nusc.get("sample_data", cam_token)
            cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"]) 
            cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
            intrinsic = np.asarray(cam_cs["camera_intrinsic"])
            
            updated_detections = []
            # Identify all 3D detections in the current camera frame and transform to 2D
            for detection_3D in frame_3D["detections"]: 
                # if "test_id" in detection_3D:
                #    breakpoint()
                if not detection_in_camera(nusc=nusc,
					   frame=frame_3D, 
                                           detection=detection_3D, 
					   camera_name=camera_pose):
                    continue
                if detection_3D["score"] > 0.1:
                    continue
                detection_3D_in_2D = transform_3D_to_2D(detection=detection_3D, 
                                                        lidar_to_global=frame_3D["lidar_to_global"], 
                                                        cam_pose=cam_pose, 
                                                        cam_cs=cam_cs)
                # Load all 2D Detections for that camera frame, 
                # TODO loading all detections on every detection again makes no sense
                detections_2D = [] 
                for detection in load_detections(path=detection_path_2D):
                    camera_token = detection["image_id"]
                    sample_camera_token_based = nusc.get("sample_data", camera_token)["sample_token"] 
                    lidar_token_2D_based = nusc.get("sample", sample_camera_token_based)["data"]["LIDAR_TOP"]
                    lidar_token_3D_based = nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]
                    
                    if detection["score"] < 0.4:
                        continue

                    if lidar_token_2D_based  == lidar_token_3D_based:
                        x, y, w, h = detection["bbox"]
                        box = BoundingBox(corners_2D=[x, y, x+w, y+h], 
                                          centers_2D=[x + (w/2), y + (h/2)], 
                                          size_2D=[w, h],
                                          label=CATEGORIES[int(detection["category_id"])],
                                          confidence_score=detection["score"],
                                          nuscenes_camera_token=camera_token
                                          )
                        detections_2D.append(box)
                
                if len(detections_2D) == 0:
                    continue
                
                # Does the current detection match any of the loaded YOLO detections? 
                matches = associate_detections(detection_3D_in_2D, detections_2D, 0.6)
                # If the matches are larger then 1, means more than one bounding box represent the same object
                # Ideally this is considered a to be a tuple of the two detections first being TF and snd YOLO
                if len(matches) > 1:
                    matches = apply_nms(matches)
                elif len(matches) == 0:
                    continue
                # If YOLO is confident here mutate the loaded transfusion detection
                # TODO clarify that this how to mutate this
                new_detection_score: float
                if matches[0][0].confidence_score >= 0.5:
                    new_detection_score = (
                        detection_3D["score"] + matches[0][0].confidence_score
                    )

                    detection_3D["score"] = new_detection_score
                    detection_3D["bbox_3d"][7] = new_detection_score
    return frames_3D
    #with open(OUTPUT_FILE, "w") as file:
    #    io.write(detections_3D, file)

if __name__ == "__main__":
    args = parse_args()
    maximise_confidence(detection_path_3D=args.detection_path_3D, 
                        version=args.version, 
                        nuscenes_root=args.nuscenes_root,
                        association_strategy=args.association_strategy
                        )
