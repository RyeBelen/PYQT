import sys
import cv2
import numpy as np
import onnxruntime as ort
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

# -------------------------------
# Helper functions for postprocessing
def non_max_suppression(boxes, iou_threshold=0.7):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[-1], reverse=True)
    final_boxes = []
    while boxes:
        chosen_box = boxes.pop(0)
        final_boxes.append(chosen_box)
        boxes = [box for box in boxes if iou(chosen_box, box) < iou_threshold]
    return final_boxes

def iou(box1, box2):
    # Each box: [x1, y1, x2, y2, label, conf]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# -------------------------------
# Worker thread for ONNX inference
class InferenceThread(QThread):
    resultReady = pyqtSignal(list)  # emit detected boxes

    def __init__(self, session, input_name, output_name, input_size, vehicle_classes, frame, original_shape, parent=None):
        super().__init__(parent)
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.input_size = input_size
        self.vehicle_classes = vehicle_classes
        self.frame = frame
        self.original_shape = original_shape

    def preprocess(self, frame):
        resized = cv2.resize(frame, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, output, original_shape):
        # Assume output is a numpy array of shape [1,84,8400]; transpose to [8400,84]
        detections = output[0].transpose()  # shape [8400,84]
        boxes = []
        for row in detections:
            conf = np.max(row[4:])  # highest class confidence
            if conf < 0.5:
                continue
            class_id = int(np.argmax(row[4:]))
            if class_id not in self.vehicle_classes:
                continue
            cx, cy, w, h = row[:4]
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            # Scale box coordinates to original frame size:
            scale_x = original_shape[1] / self.input_size[0]
            scale_y = original_shape[0] / self.input_size[1]
            boxes.append([x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y,
                          self.vehicle_classes[class_id], conf])
        return non_max_suppression(boxes, iou_threshold=0.7)

    def run(self):
        inp = self.preprocess(self.frame)
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        boxes = self.postprocess(outputs, self.original_shape)
        self.resultReady.emit(boxes)

# -------------------------------
class VideoWidget(QLabel):
    def __init__(self, video_path, onnx_model_path, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(video_path)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(30)  # ~30ms interval

        # Create an ONNX session only once.
        providers = ['CPUExecutionProvider']  # Change to 'CUDAExecutionProvider' if available
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (640, 640)
        self.vehicle_classes = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

        self.frame_skip = 1
        self.counter = 0
        self.inference_thread = None  # Store the active inference thread

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        self.counter = (self.counter + 1) % self.frame_skip
        if self.counter != 0:
            self.displayFrame(frame)
            return

        # If a previous thread is still running, skip this frame.
        if self.inference_thread is not None and self.inference_thread.isRunning():
            self.displayFrame(frame)
            return

        # Create and start a new inference thread.
        self.inference_thread = InferenceThread(
            self.session, self.input_name, self.output_name,
            self.input_size, self.vehicle_classes, frame.copy(), frame.shape
        )
        self.inference_thread.resultReady.connect(lambda boxes: self.drawAndDisplay(frame, boxes))
        # When finished, delete the thread.
        self.inference_thread.finished.connect(self.cleanupThread)
        self.inference_thread.start()

    def cleanupThread(self):
        if self.inference_thread is not None:
            self.inference_thread.deleteLater()
            self.inference_thread = None

    def drawAndDisplay(self, frame, boxes):
        for box in boxes:
            x1, y1, x2, y2, label, conf = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{label}:{conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        self.displayFrame(frame)

    def displayFrame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        # Ensure the inference thread is finished before closing.
        if self.inference_thread is not None and self.inference_thread.isRunning():
            self.inference_thread.quit()
            self.inference_thread.wait()
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self, video_path, onnx_model_path):
        super().__init__()
        self.setWindowTitle("YOLOv8 Vehicle Detection")
        self.widget = VideoWidget(video_path, onnx_model_path)
        self.setCentralWidget(self.widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_path = "Prototypes/Vids/testvid_1.mp4"          # Path to your prerecorded video file
    onnx_model_path = "Prototypes/Models/TRAFFICO_v1.onnx"  # Your ONNX model file path
    mainWin = MainWindow(video_path, onnx_model_path)
    mainWin.resize(800, 600)
    mainWin.show()
    sys.exit(app.exec_())
