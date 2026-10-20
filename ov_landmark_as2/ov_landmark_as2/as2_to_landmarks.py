"""Aerostack2 ObjectPerceptionArray -> ov_landmark_msgs/LandmarkDetectionArray, for the OpenVINS map landmark update.

mode "aruco":   class_id is the marker id (as2 aruco plugin), the detection type is "aruco"
mode "objects": class_id is the landmark object type (e.g. YOLO gate_blue), the id is unknown (-1)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from as2_msgs.msg import ObjectPerceptionArray
from ov_landmark_msgs.msg import LandmarkDetection, LandmarkDetectionArray, LandmarkKeypoint


class As2ToLandmarks(Node):
    def __init__(self):
        super().__init__('as2_to_landmarks')
        self.mode = self.declare_parameter('mode', 'aruco').value
        if self.mode not in ('aruco', 'objects'):
            raise ValueError(f'mode must be aruco or objects, got {self.mode}')
        input_topic = self.declare_parameter('input_topic', '/drone0/detections/aruco/front').value
        output_topic = self.declare_parameter('output_topic', '/ov_msckf/landmarks/front').value
        self.pub = self.create_publisher(LandmarkDetectionArray, output_topic, qos_profile_sensor_data)
        self.create_subscription(ObjectPerceptionArray, input_topic, self.callback, qos_profile_sensor_data)
        self.get_logger().info(f'{self.mode}: {input_topic} -> {output_topic}')

    def callback(self, msg):
        out = LandmarkDetectionArray()
        out.header = msg.header
        for p in msg.perceptions:
            det = LandmarkDetection()
            class_id = p.hypothesis.hypothesis.class_id
            if self.mode == 'aruco':
                if not class_id.isdigit():
                    continue
                det.type, det.id = 'aruco', int(class_id)
            else:
                det.type, det.id = class_id, -1
            for k, kp in enumerate(p.keypoints):
                score = p.keypoint_scores[k] if k < len(p.keypoint_scores) else 1.0
                det.keypoints.append(LandmarkKeypoint(x=float(kp.x), y=float(kp.y), score=float(score)))
            det.box_area = float(p.bbox.size_x * p.bbox.size_y)
            out.detections.append(det)
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(As2ToLandmarks())


if __name__ == '__main__':
    main()
