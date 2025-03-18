from random import randint
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Temperature vs time dynamic plot
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")
        self.pen = pg.mkPen(color=(255, 0, 0))
        self.plot_graph.setTitle("Temperature vs Time", color="b", size="20pt")
        styles = {"color": "red", "font-size": "18px"}
        self.plot_graph.setLabel("left", "Temperature (°C)", **styles)
        self.plot_graph.setLabel("bottom", "Time (min)", **styles)
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(20, 40)
        
        # Initial data
        self.time = list(range(10))
        self.temperature = [randint(20, 40) for _ in range(10)]
        
        # Get a line reference for the main data
        self.line = self.plot_graph.plot(
            self.time,
            self.temperature,
            name="Temperature Sensor",
            pen=self.pen,
            symbol="+",
            symbolSize=15,
            symbolBrush="b",
        )
        
        # Add a separate line for the animated segment
        self.animated_segment = self.plot_graph.plot(
            [], [], 
            pen=self.pen
        )
        
        # Animation parameters
        self.animation_frames = 15  # Number of frames for smooth animation
        self.frame_counter = 0
        self.new_point_x = None
        self.new_point_y = None
        self.previous_point_x = None
        self.previous_point_y = None
        self.is_animating = False
        
        # Add a timer for animation frames
        self.timer = QtCore.QTimer()
        self.timer.setInterval(30)  # Animation update interval
        self.timer.timeout.connect(self.animate)
        self.timer.start()
        
        # Add a timer to generate new data points
        self.data_timer = QtCore.QTimer()
        self.data_timer.setInterval(1500)  # New data point every 1.5 seconds
        self.data_timer.timeout.connect(self.update_data)
        self.data_timer.start()
    
    def update_data(self):
        """Generate a new data point and start animation"""
        if self.is_animating:
            return  # Don't add new points during animation
            
        # Store the previous endpoint
        self.previous_point_x = self.time[-1]
        self.previous_point_y = self.temperature[-1]
        
        # Remove the oldest data point
        self.time = self.time[1:]
        self.temperature = self.temperature[1:]
        
        # Generate new endpoint
        new_time = self.previous_point_x + 1
        new_temp = randint(20, 40)
        
        # Store the new endpoint (but don't add to main line yet)
        self.new_point_x = new_time
        self.new_point_y = new_temp
        
        # Update the plot without the new point
        self.line.setData(self.time, self.temperature)
        
        # Start the animation
        self.frame_counter = 0
        self.is_animating = True
    
    def animate(self):
        """Animate the line extending to the new point"""
        if not self.is_animating:
            return
            
        if self.frame_counter < self.animation_frames:
            # Calculate how far along the animation we are (0.0 to 1.0)
            progress = self.frame_counter / self.animation_frames
            
            # Calculate the current position along the line
            current_x = self.previous_point_x + (self.new_point_x - self.previous_point_x) * progress
            current_y = self.previous_point_y + (self.new_point_y - self.previous_point_y) * progress
            
            # Draw the animated line segment
            self.animated_segment.setData(
                [self.previous_point_x, current_x],
                [self.previous_point_y, current_y]
            )
            
            self.frame_counter += 1
        else:
            # Animation complete, add the new point to the main dataset
            self.time.append(self.new_point_x)
            self.temperature.append(self.new_point_y)
            self.line.setData(self.time, self.temperature)
            
            # Clear the animation line
            self.animated_segment.setData([], [])
            self.is_animating = False

app = QtWidgets.QApplication([])
main = MainWindow()
main.show()
app.exec()