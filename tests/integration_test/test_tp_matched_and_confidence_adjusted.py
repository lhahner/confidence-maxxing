"""
Generally confidence maxxing should identify low confidence true positive detections
coming from TransFusion. In sample f4f86af4da3b49e79497deda5c5f223a there are
three objects with a detection score below 0.05, these detections should be detected
by YOLO and thus matched with the transfusion detections to increase finally the
confidence score. We specifically look at sample 42563f19747741318b4983bbc61a47b1 
from the CAM_FRONT sensor there is one car which transfusions assigns a confidence
of raphly 0.05 and YOLO more then 0.5. This test should verify that the
method finds the detection and mutates its detections score.

E.g. the detection:
 {'label': 'car',
     'score': 0.05057337135076523,
     'bbox_3d': [-11.649444580078125,
     	      45.278099060058594,
     	       2.4196410179138184,
     	      -2.900900363922119,
     	       4.642308712005615,
     	       2.0160934925079346,
     	       1.8797235488891602,
     	       0.05057337135076523
 	    ]
 'test_id': "31082026"
 }

To oberserve that the actual expected target item has changed I have added
an id to the detected object in the test data, to making calling at the
beginning and calling ad the end easier.
"""
import unittest
import json

from main import maximise_confidence

class TestConfidenceMaxxing(unittest.TestCase):
 def find_TP_detections(self, data):
            tp_low_confidence_object_detection = {} 
            for data in data["frames"]:
                detections = data["detections"]
                for detection in detections:
                    if "test_id" in detection:
                       tp_low_confidence_object_detection.update(detection)
            return tp_low_confidence_object_detection
 	
 def test_TP_with_low_confidence_is_identified_and_score_is_updated(self):
     with open("/user/lennart.hahner/u28856/confidence-maxxing/tests/data/transfusion_test_sample.json", "r") as f:
                data = json.load(f)
     score_before = self.find_TP_detections(data)["score"]
     mutated_object = maximise_confidence(detection_path_3D="/user/lennart.hahner/u28856/confidence-maxxing/tests/data/transfusion_test_sample.json",
                                                 detection_path_2D="/user/lennart.hahner/u28856/confidence-maxxing/tests/data/yolo_test_sample.json",
                                                 version="v1.0-mini",
                                                 association_strategy="iou-based")
     score_after = self.find_TP_detections(mutated_object)["score"]
     self.assertTrue(float(score_before) < float(score_after))
