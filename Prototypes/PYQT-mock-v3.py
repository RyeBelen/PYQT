import sys
import cv2
import numpy as np
import onnxruntime
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
                           QWidget, QFileDialog, QHBoxLayout, QLineEdit, QGroupBox, QSlider,
                           QComboBox, QCheckBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# Optional import for PyTorch - will be checked at runtime
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class VideoProcessingThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, list)
    processing_fps = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_path, video_path, skip_frames=2):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.running = False
        self.skip_frames = skip_frames
        self.conf_threshold = 0.5
        
        # Only track vehicle classes
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck indices
        self.class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        
        self.model_type = 'onnx' if model_path.endswith('.onnx') else 'pt'
        self.model = None
    
    def initialize_model(self):
        try:
            if self.model_type == 'onnx':
                # Initialize the ONNX Runtime session with optimizations
                sess_options = onnxruntime.SessionOptions()
                sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.enable_mem_pattern = True
                sess_options.enable_cpu_mem_arena = True
                
                # Initialize with preferred execution providers
                providers = ['CPUExecutionProvider']
                if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
                    
                self.model = onnxruntime.InferenceSession(
                    self.model_path, 
                    sess_options=sess_options,
                    providers=providers
                )
                
                self.input_name = self.model.get_inputs()[0].name
                self.output_names = [output.name for output in self.model.get_outputs()]
                
                # Get input shape
                input_shape = self.model.get_inputs()[0].shape
                self.input_width = input_shape[3] 
                self.input_height = input_shape[2]
                
            elif self.model_type == 'pt':
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch is not installed. Cannot load .pt model.")
                
                # Import here to avoid dependency if not needed
                from ultralytics import YOLO
                
                # Load the YOLOv8 model
                self.model = YOLO(self.model_path)
                
                # Set input dimensions based on model configuration
                self.input_width = self.input_height = 640  # Default YOLOv8 input size
                
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Model initialization error: {str(e)}")
            return False
    
    def run(self):
        if not self.initialize_model():
            return
            
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            self.error_occurred.emit(f"Could not open video: {self.video_path}")
            return
            
        frame_count = 0
        self.running = True
        
        processing_times = []
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames to improve performance
            frame_count += 1
            if frame_count % (self.skip_frames + 1) != 0:
                continue
            
            start_time = time.time()
            
            # Detect vehicles
            detections = self.detect_vehicles(frame)
            
            # Calculate processing FPS
            process_time = time.time() - start_time
            processing_times.append(process_time)
            
            # Calculate rolling average FPS (last 10 frames)
            if len(processing_times) > 10:
                processing_times.pop(0)
            avg_process_time = sum(processing_times) / len(processing_times)
            fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
            self.processing_fps.emit(fps)
            
            # Emit processed frame and detections
            self.frame_ready.emit(frame, detections)
            
        cap.release()
    
    def detect_vehicles(self, frame):
        if self.model_type == 'onnx':
            return self.detect_vehicles_onnx(frame)
        else:  # pt model
            return self.detect_vehicles_pt(frame)
    
    def detect_vehicles_onnx(self, frame):
        # Resize image while preserving aspect ratio
        img = frame.copy()
        h, w = img.shape[:2]
        
        # Calculate scale
        scale = min(self.input_width / w, self.input_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize and pad with border
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        pad_img = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        pad_img[:new_h, :new_w, :] = resized
        
        # Preprocess (normalize and convert to NCHW)
        input_img = pad_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_img, axis=0)
        
        # Run inference
        outputs = self.model.run(self.output_names, {self.input_name: input_tensor})
        
        # Process predictions (YOLOv8 format)
        boxes = []
        predictions = outputs[0]
        
        for prediction in predictions:
            for i in range(len(prediction)):
                row = prediction[i]
                confidence = row[4]
                
                if confidence < self.conf_threshold:
                    continue
                
                # Get class scores
                class_scores = row[5:]
                class_id = np.argmax(class_scores)
                
                # Only keep vehicle detections
                if class_id in self.vehicle_classes:
                    # Get box in xywh format (center_x, center_y, width, height)
                    x, y, w_box, h_box = row[0], row[1], row[2], row[3]
                    
                    # Convert normalized coordinates back to original image scale
                    ratio_w, ratio_h = w / new_w, h / new_h
                    
                    # Convert to corner format (x1, y1, x2, y2)
                    x1 = int((x - w_box/2) * new_w * ratio_w)
                    y1 = int((y - h_box/2) * new_h * ratio_h)
                    x2 = int((x + w_box/2) * new_w * ratio_w)
                    y2 = int((y + h_box/2) * new_h * ratio_h)
                    
                    # Add to boxes
                    boxes.append([x1, y1, x2, y2, confidence, class_id])
        
        return boxes
    
    def detect_vehicles_pt(self, frame):
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                
                # Only keep vehicle detections
                if cls_id in self.vehicle_classes:
                    conf = float(box.conf.item())
                    xyxy = box.xyxy[0].tolist()  # Get box in [x1, y1, x2, y2] format
                    
                    x1, y1, x2, y2 = map(int, xyxy)
                    boxes.append([x1, y1, x2, y2, conf, cls_id])
        
        return boxes
    
    def stop(self):
        self.running = False
        self.wait()


class VehicleDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.video_path = None
        self.model_path = None
        self.processing_thread = None
        self.skip_frames = 2  # Default frame skipping
        
        # Check for PyTorch availability
        self.torch_available = TORCH_AVAILABLE
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("Vehicle Detection - Multi-format Support")
        self.setGeometry(100, 100, 900, 700)
        
        # Create main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        # Path Configuration Group
        path_group = QGroupBox("Configuration")
        path_layout = QVBoxLayout()
        
        # Model type info
        model_type_layout = QHBoxLayout()
        model_type_layout.addWidget(QLabel("Supported Model Types:"))
        self.torch_status = QLabel("PyTorch (.pt): " + ("Available ✓" if self.torch_available else "Not Available ✗"))
        self.torch_status.setStyleSheet("color: " + ("green" if self.torch_available else "red"))
        model_type_layout.addWidget(self.torch_status)
        
        self.onnx_status = QLabel("ONNX (.onnx): Available ✓")
        self.onnx_status.setStyleSheet("color: green")
        model_type_layout.addWidget(self.onnx_status)
        
        path_layout.addLayout(model_type_layout)
        
        # Model path selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Path:"))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to YOLOv8 model (.pt or .onnx)")
        model_layout.addWidget(self.model_path_edit)
        self.model_browse_button = QPushButton("Browse")
        self.model_browse_button.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_browse_button)
        path_layout.addLayout(model_layout)
        
        # Video path selection
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("Video Path:"))
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Path to pre-recorded video file")
        video_layout.addWidget(self.video_path_edit)
        self.video_browse_button = QPushButton("Browse")
        self.video_browse_button.clicked.connect(self.browse_video)
        video_layout.addWidget(self.video_browse_button)
        path_layout.addLayout(video_layout)
        
        # Performance settings
        perf_layout = QHBoxLayout()
        
        # Frame skip slider
        perf_layout.addWidget(QLabel("Frame Skip:"))
        self.skip_slider = QSlider(Qt.Horizontal)
        self.skip_slider.setMinimum(0)
        self.skip_slider.setMaximum(10)
        self.skip_slider.setValue(self.skip_frames)
        self.skip_slider.setTickPosition(QSlider.TicksBelow)
        self.skip_slider.setTickInterval(1)
        self.skip_slider.valueChanged.connect(self.update_skip_frames)
        perf_layout.addWidget(self.skip_slider)
        
        self.skip_label = QLabel(f"Skip: {self.skip_frames} frames")
        perf_layout.addWidget(self.skip_label)
        
        path_layout.addLayout(perf_layout)
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # Video display
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f0f0f0;")
        layout.addWidget(self.display_label)
        
        # Performance indicators
        perf_indicator_layout = QHBoxLayout()
        self.fps_label = QLabel("Processing FPS: -")
        perf_indicator_layout.addWidget(self.fps_label)
        self.vehicle_count_label = QLabel("Vehicles: -")
        perf_indicator_layout.addWidget(self.vehicle_count_label)
        self.model_type_label = QLabel("Model Type: -")
        perf_indicator_layout.addWidget(self.model_type_label)
        layout.addLayout(perf_indicator_layout)
        
        # Status label
        self.status_label = QLabel("Please select model and video files")
        self.status_label.setStyleSheet("font-weight: bold; color: #555555;")
        layout.addWidget(self.status_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        # Set the layout
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
    
    def update_skip_frames(self, value):
        self.skip_frames = value
        self.skip_label.setText(f"Skip: {value} frames")
        if self.processing_thread:
            self.processing_thread.skip_frames = value
    
    def browse_model(self):
        file_filter = "YOLOv8 Models (*.pt *.onnx)"
        if not self.torch_available:
            file_filter = "ONNX Files (*.onnx)"
            
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(
            self, "Select YOLOv8 Model", "", file_filter
        )
        
        if model_path:
            if not self.torch_available and model_path.endswith('.pt'):
                self.status_label.setText("Error: PyTorch is not installed. Cannot use .pt models.")
                return
                
            self.model_path = model_path
            self.model_path_edit.setText(model_path)
            
            model_type = "PyTorch" if model_path.endswith('.pt') else "ONNX"
            self.model_type_label.setText(f"Model Type: {model_type}")
            
            self.check_ready_state()
    
    def browse_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        
        if video_path:
            self.video_path = video_path
            self.video_path_edit.setText(video_path)
            self.check_ready_state()
    
    def check_ready_state(self):
        if self.model_path and self.video_path:
            self.start_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)
    
    def update_fps(self, fps):
        self.fps_label.setText(f"Processing FPS: {fps:.1f}")
    
    def update_display(self, frame, detections):
        # Draw bounding boxes
        frame_with_boxes = self.draw_boxes(frame, detections)
        
        # Update vehicle count
        self.vehicle_count_label.setText(f"Vehicles: {len(detections)}")
        
        # Convert to QImage and display
        h, w, ch = frame_with_boxes.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(frame_with_boxes.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        self.display_label.setPixmap(pixmap.scaled(self.display_label.width(), self.display_label.height(), Qt.KeepAspectRatio))
    
    def handle_error(self, error_msg):
        self.status_label.setText(f"Error: {error_msg}")
        self.stop_detection()
    
    def start_detection(self):
        # Update paths from text fields (in case user typed them)
        self.model_path = self.model_path_edit.text()
        self.video_path = self.video_path_edit.text()
        
        if not os.path.exists(self.model_path):
            self.status_label.setText("Model file not found. Please check the path.")
            return
            
        if not os.path.exists(self.video_path):
            self.status_label.setText("Video file not found. Please check the path.")
            return
            
        # Check file extensions
        if not self.model_path.endswith(('.pt', '.onnx')):
            self.status_label.setText("Model must be a .pt or .onnx file.")
            return
            
        if self.model_path.endswith('.pt') and not self.torch_available:
            self.status_label.setText("PyTorch is not installed. Cannot use .pt models.")
            return
        
        # Create and start the processing thread
        self.processing_thread = VideoProcessingThread(self.model_path, self.video_path, self.skip_frames)
        self.processing_thread.frame_ready.connect(self.update_display)
        self.processing_thread.processing_fps.connect(self.update_fps)
        self.processing_thread.error_occurred.connect(self.handle_error)
        self.processing_thread.start()
        
        video_filename = os.path.basename(self.video_path)
        model_filename = os.path.basename(self.model_path)
        model_type = "PyTorch" if self.model_path.endswith('.pt') else "ONNX"
        
        self.status_label.setText(f"Detection started on: {video_filename} using {model_type} model: {model_filename}")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.model_browse_button.setEnabled(False)
        self.video_browse_button.setEnabled(False)
    
    def stop_detection(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
        
        self.status_label.setText("Detection stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.model_browse_button.setEnabled(True)
        self.video_browse_button.setEnabled(True)
    
    def draw_boxes(self, frame, detections):
        # Make a copy to avoid modifying the original frame
        output_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for box in detections:
            x1, y1, x2, y2, conf, class_id = box
            
            # Ensure coordinates are within frame boundaries
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            
            # Get class name
            class_name = self.processing_thread.class_names[int(class_id)]
            label = f"{class_name}: {conf:.2f}"
            
            # Draw box (faster rectangle drawing)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label (simplified)
            cv2.putText(output_img, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add vehicle count (simplified)
        count_text = f"Count: {len(detections)}"
        cv2.putText(output_img, count_text, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        return output_img

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VehicleDetectionApp()
    window.show()
    sys.exit(app.exec_())