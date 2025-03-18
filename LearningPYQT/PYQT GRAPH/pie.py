import sys
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRectF, pyqtProperty

class PieSlice:
    def __init__(self, value=0, color=None):
        self._value = value
        self._target_value = value
        self._color = color if color else QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
    @property
    def value(self):
        return self._value
        
    @value.setter
    def value(self, val):
        self._value = val
        
    @property
    def target_value(self):
        return self._target_value
        
    @target_value.setter
    def target_value(self, val):
        self._target_value = val
        
    @property
    def color(self):
        return self._color


class PieChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.slices = [PieSlice(random.randint(1, 10)) for _ in range(5)]
        self.total_value = sum(slice.value for slice in self.slices)
        
        self._animation_progress = 0.0
        self._size_factor = 1.0
        self._target_size_factor = 1.0
        
        # Animation timer for smooth transitions
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(16)  # ~60 FPS
        
        # Size animation for bouncy effect
        self.size_animation = QPropertyAnimation(self, b"size_factor")
        self.size_animation.setDuration(500)
        self.size_animation.setEasingCurve(QEasingCurve.OutElastic)
        
        # Auto-change timer - NEW
        self.auto_change_timer = QTimer(self)
        self.auto_change_timer.timeout.connect(self.randomize_values)
        self.auto_change_timer.start(3000)  # Change every 3 seconds
    
    def get_size_factor(self):
        return self._size_factor
        
    def set_size_factor(self, factor):
        self._size_factor = factor
        self.update()
        
    size_factor = pyqtProperty(float, get_size_factor, set_size_factor)
    
    def update_animation(self):
        animating = False
        for slice in self.slices:
            if abs(slice.value - slice.target_value) > 0.01:
                slice.value += (slice.target_value - slice.value) * 0.1
                animating = True
                
        if abs(self._size_factor - self._target_size_factor) > 0.01:
            #self._size_factor += (self._target_size_factor - self._size_factor) * 0.1
            animating = True
            
        self.total_value = sum(slice.value for slice in self.slices)
        
        if animating:
            self.update()
    
    def randomize_values(self):
        for slice in self.slices:
            slice.target_value = random.randint(1, 10)
            
        # Add a size bounce effect when values change
        #self._target_size_factor = 1.0
        #self.size_animation.setStartValue(0.9)  # Start slightly smaller
        self.size_animation.setEndValue(self._target_size_factor)
        self.size_animation.start()
    
    def paintEvent(self, event):
        if self.total_value <= 0:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate size and position
        width = self.width() * self._size_factor
        height = self.height() * self._size_factor
        
        # Center the pie chart
        x = (self.width() - width) / 2
        y = (self.height() - height) / 2
        
        # Draw the pie chart
        rect = QRectF(x, y, width, height)
        start_angle = 0
        
        for slice in self.slices:
            # Calculate the span angle for this slice
            span_angle = (slice.value / self.total_value) * 360 * 16  # QPainter uses 16ths of a degree
            
            # Draw the slice
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(slice.color))
            painter.drawPie(rect, int(start_angle), int(span_angle))
            
            # Update start angle for next slice
            start_angle += span_angle


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto-Changing Pie Chart")
        self.setGeometry(100, 100, 500, 500)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.pie_chart = PieChart()
        layout.addWidget(self.pie_chart)
        
        button_layout = QHBoxLayout()
        randomize_button = QPushButton("Randomize Now")
        randomize_button.clicked.connect(self.pie_chart.randomize_values)
        button_layout.addWidget(randomize_button)
        
        # Add a button to toggle auto-changing - NEW
        self.toggle_button = QPushButton("Stop Auto-Change")
        self.toggle_button.clicked.connect(self.toggle_auto_change)
        button_layout.addWidget(self.toggle_button)
        
        layout.addLayout(button_layout)
    
    # Toggle auto-change function - NEW
    def toggle_auto_change(self):
        if self.pie_chart.auto_change_timer.isActive():
            self.pie_chart.auto_change_timer.stop()
            self.toggle_button.setText("Start Auto-Change")
        else:
            self.pie_chart.auto_change_timer.start()
            self.toggle_button.setText("Stop Auto-Change")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())