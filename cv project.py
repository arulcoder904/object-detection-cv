import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading


class YOLOObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")

        # Variables
        self.cap = None
        self.running = False
        self.whT = 320
        self.confThreshold = 0.5
        self.nmsThreshold = 0.3

        # Colors for different objects
        self.detection_color = (0, 255, 0)  # Green for detection box

        # Load YOLO model
        self.classNames = []
        self.net = None
        self.load_yolo_model()

        # Create GUI
        self.create_widgets()

    def load_yolo_model(self):
        try:
            classesfile = 'coco.names'
            with open(classesfile, 'rt') as f:
                self.classNames = f.read().rstrip('\n').split('\n')

            modelConfig = 'yolov3.cfg'
            modelWeights = 'yolov3.weights'
            self.net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {str(e)}")

    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel (controls)
        control_frame = tk.Frame(main_frame, bg="#34495e", bd=2, relief=tk.RAISED)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Title label
        title_label = tk.Label(control_frame, text="Object Detection System",
                               font=("Helvetica", 16, "bold"), bg="#34495e", fg="#ecf0f1")
        title_label.pack(pady=20)

        # Start/Stop buttons
        button_frame = tk.Frame(control_frame, bg="#34495e")
        button_frame.pack(pady=20)

        self.start_button = tk.Button(button_frame, text="Start Camera", command=self.start_camera,
                                      bg="#27ae60", fg="white", font=("Helvetica", 12, "bold"),
                                      width=15, relief=tk.RAISED, bd=3)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop Camera", command=self.stop_camera,
                                     bg="#e74c3c", fg="white", font=("Helvetica", 12, "bold"),
                                     width=15, relief=tk.RAISED, bd=3, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Settings frame
        settings_frame = tk.LabelFrame(control_frame, text="Detection Settings",
                                       font=("Helvetica", 12), bg="#34495e", fg="#ecf0f1")
        settings_frame.pack(pady=20, padx=10, fill=tk.X)

        # Confidence threshold slider
        conf_label = tk.Label(settings_frame, text="Confidence Threshold:",
                              bg="#34495e", fg="#ecf0f1")
        conf_label.pack(anchor=tk.W, pady=(5, 0))

        self.conf_slider = tk.Scale(settings_frame, from_=0.1, to=1.0, resolution=0.05,
                                    orient=tk.HORIZONTAL, bg="#34495e", fg="#ecf0f1",
                                    highlightthickness=0, troughcolor="#7f8c8d",
                                    activebackground="#3498db", length=200)
        self.conf_slider.set(self.confThreshold)
        self.conf_slider.pack(pady=5)

        # NMS threshold slider
        nms_label = tk.Label(settings_frame, text="NMS Threshold:",
                             bg="#34495e", fg="#ecf0f1")
        nms_label.pack(anchor=tk.W, pady=(5, 0))

        self.nms_slider = tk.Scale(settings_frame, from_=0.1, to=0.5, resolution=0.05,
                                   orient=tk.HORIZONTAL, bg="#34495e", fg="#ecf0f1",
                                   highlightthickness=0, troughcolor="#7f8c8d",
                                   activebackground="#3498db", length=200)
        self.nms_slider.set(self.nmsThreshold)
        self.nms_slider.pack(pady=5)

        # Detection info frame
        info_frame = tk.LabelFrame(control_frame, text="Detection Information",
                                   font=("Helvetica", 12), bg="#34495e", fg="#ecf0f1")
        info_frame.pack(pady=20, padx=10, fill=tk.BOTH, expand=True)

        self.detection_text = tk.Text(info_frame, height=12, width=35,
                                      bg="#2c3e50", fg="#ecf0f1", wrap=tk.WORD,
                                      font=("Courier", 10))
        self.detection_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

        # Object counter
        counter_frame = tk.Frame(info_frame, bg="#34495e")
        counter_frame.pack(fill=tk.X, pady=5)

        tk.Label(counter_frame, text="Objects Detected:", bg="#34495e", fg="#ecf0f1",
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.object_count_label = tk.Label(counter_frame, text="0", bg="#34495e", fg="#3498db",
                                           font=("Helvetica", 12, "bold"))
        self.object_count_label.pack(side=tk.RIGHT)

        # Right panel (video display)
        video_frame = tk.Frame(main_frame, bg="#34495e", bd=2, relief=tk.RAISED)
        video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_label = tk.Label(video_frame, bg="#2c3e50")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video device")
                return

            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
