import sys
import cv2
import numpy as np
import onnxruntime as ort
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout

# Helper functions for processing YOLOv8 ONNX output
def non_max_suppression(boxes, iou_threshold=0.7):
    if len(boxes) == 0:
        return []
    # sort by confidence score (last element)
    boxes = sorted(boxes, key=lambda x: x[-1], reverse=True)
    final_boxes = []
    while boxes:
        chosen_box = boxes.pop(0)
        final_boxes.append(chosen_box)
        boxes = [
            box for box in boxes 
            if iou(chosen_box, box) < iou_threshold
        ]
    return final_boxes

def iou(box1, box2):
    # each box: [x1,y1,x2,y2, class, conf]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

class VideoWidget(QLabel):
    def __init__(self, video_path, onnx_model_path, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(video_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(30)  # ~33ms for ~30 FPS

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        # Name of model input (usually "images") and output (e.g. "output0")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # For demo, assume input shape is [1,3,640,640]
        self.input_size = (640, 640)
        # Simple mapping for COCO: we'll assume class 2 is "car", 3 is "motorbike", 5 is "bus", 7 is "truck"
        self.vehicle_classes = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

    def preprocess(self, frame):
        # Resize to model's expected size and convert BGR to RGB
        resized = cv2.resize(frame, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0,1] and transpose to [1,3,H,W]
        img = rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, output, original_shape, resized_shape):
        # Here we assume output is a numpy array of shape [1,84,8400]
        # Each detection row: first 4 values: center_x, center_y, width, height (relative to resized image)
        # Next 80 values: class confidences.
        # We'll convert center-based coords to xyxy format.
        detections = output[0]  # shape [84,8400]
        detections = detections.transpose()  # shape [8400,84]
        boxes = []
        for row in detections:
            conf = np.max(row[4:])  # highest class confidence
            if conf < 0.5:
                continue
            class_id = int(np.argmax(row[4:]))
            if class_id not in self.vehicle_classes:
                continue
            cx, cy, w, h = row[:4]
            # Convert center-format to top-left and bottom-right on resized image:
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            # Scale box coordinates to original frame size:
            scale_x = original_shape[1] / resized_shape[0]
            scale_y = original_shape[0] / resized_shape[1]
            boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y,
                          self.vehicle_classes[class_id], conf])
        # Apply non-max suppression:
        boxes = non_max_suppression(boxes, iou_threshold=0.7)
        return boxes

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        input_img = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_img})
        boxes = self.postprocess(outputs, frame.shape, self.input_size)

        # Draw boxes on the frame:
        for box in boxes:
            x1, y1, x2, y2, label, conf = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{label}:{conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Convert BGR frame to RGB QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

class MainWindow(QMainWindow):
    def __init__(self, video_path, onnx_model_path):
        super().__init__()
        self.setWindowTitle("YOLOv8 Vehicle Detection")
        self.widget = VideoWidget(video_path, onnx_model_path)
        self.setCentralWidget(self.widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_path = "Prototypes/Vids/testvid_1.mp4"          # Path to your prerecorded video file
    onnx_model_path = "Prototypes/Models/TRAFFICO_v1.onnx"  # Path to your ONNX exported YOLOv8 model
    mainWin = MainWindow(video_path, onnx_model_path)
    mainWin.resize(800,600)
    mainWin.show()
    sys.exit(app.exec_())
