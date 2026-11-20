import json
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


ROOT = os.path.dirname(os.path.abspath(__file__))
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Studio")
        self.root.geometry("1120x720")
        self.root.minsize(900, 600)

        self.running = False
        self.registration = False
        self.capture_count = 0
        self.registration_name = ""
        self.registration_folder = ""
        self.capture = None
        self.recognizer = None
        self.names = {}
        self.emotion_model = None
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.photo = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        self.root.configure(bg="#101827")
        shell = tk.Frame(self.root, bg="#101827")
        shell.pack(fill="both", expand=True)

        sidebar = tk.Frame(shell, bg="#172238", width=260)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="FACE STUDIO",
            fg="#ffffff",
            bg="#172238",
            font=("Segoe UI", 20, "bold"),
        ).pack(pady=(24, 4), padx=20, anchor="w")
        tk.Label(
            sidebar,
            text="Recognition and expression tools",
            fg="#91a4c7",
            bg="#172238",
            font=("Segoe UI", 9),
        ).pack(padx=20, anchor="w")

        self.status_dot = tk.Canvas(sidebar, width=12, height=12, highlightthickness=0, bg="#172238")
        self.status_dot.place(x=20, y=96)
        self.status_dot.create_oval(2, 2, 10, 10, fill="#ef4444", outline="")
        self.status_label = tk.Label(
            sidebar, text="Camera stopped", fg="#d8e1f0", bg="#172238", font=("Segoe UI", 10)
        )
        self.status_label.place(x=40, y=91)

        controls = tk.LabelFrame(
            sidebar, text="Live camera", fg="#aebdd5", bg="#172238", bd=1,
            relief="solid", labelanchor="n"
        )
        controls.pack(fill="x", padx=18, pady=(110, 12))

        self.start_button = ttk.Button(controls, text="Start Recognition", command=self.toggle_camera)
        self.start_button.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Button(controls, text="Open Image", command=self.open_image).pack(fill="x", padx=10, pady=6)

        register = tk.LabelFrame(
            sidebar, text="Identity enrollment", fg="#aebdd5", bg="#172238", bd=1,
            relief="solid", labelanchor="n"
        )
        register.pack(fill="x", padx=18, pady=12)

        tk.Label(register, text="Participant name", fg="#d8e1f0", bg="#172238").pack(
            anchor="w", padx=10, pady=(8, 2)
        )
        self.name_entry = ttk.Entry(register)
        self.name_entry.pack(fill="x", padx=10, pady=(0, 8))
        self.register_button = ttk.Button(register, text="Start 60 Captures", command=self.toggle_registration)
        self.register_button.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Button(register, text="Train Identity Model", command=self.train_identity).pack(
            fill="x", padx=10, pady=6
        )

        train = tk.LabelFrame(
            sidebar, text="Expression model", fg="#aebdd5", bg="#172238", bd=1,
            relief="solid", labelanchor="n"
        )
        train.pack(fill="x", padx=18, pady=12)
        self.epochs = tk.StringVar(value="25")
        tk.Label(train, text="Epochs", fg="#d8e1f0", bg="#172238").pack(anchor="w", padx=10, pady=(8, 2))
        tk.Spinbox(train, from_=1, to=100, textvariable=self.epochs, width=8).pack(
            anchor="w", padx=10, pady=(0, 8)
        )
        ttk.Button(train, text="Train Emotion Model", command=self.train_emotion).pack(
            fill="x", padx=10, pady=(0, 10)
        )

        main = tk.Frame(shell, bg="#101827")
        main.pack(side="left", fill="both", expand=True)
        top = tk.Frame(main, bg="#101827")
        top.pack(fill="x", padx=18, pady=(18, 10))
        tk.Label(top, text="Live Preview", fg="#ffffff", bg="#101827", font=("Segoe UI", 15, "bold")).pack(side="left")
        tk.Label(top, text="LBPH threshold: 75", fg="#91a4c7", bg="#101827").pack(side="right")

        preview = tk.Frame(main, bg="#0b1220", bd=1, relief="solid")
        preview.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        self.canvas = tk.Canvas(preview, bg="#0b1220", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.canvas.delete("placeholder"))
        self.canvas.create_text(
            400, 250, text="Camera preview", fill="#53627d",
            font=("Segoe UI", 16), tags="placeholder"
        )

        self.log = tk.Text(
            main, height=7, bg="#080d16", fg="#b8c4d9", insertbackground="#b8c4d9",
            relief="flat", padx=12, pady=8, font=("Consolas", 9)
        )
        self.log.pack(fill="x", padx=18, pady=(0, 18))
        self.log_message("Ready. Models are loaded when recognition starts.")

    def log_message(self, message):
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def set_camera_status(self, active, text):
        color = "#22c55e" if active else "#ef4444"
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 10, 10, fill=color, outline="")
        self.status_label.configure(text=text)

    def load_models(self):
        if not hasattr(cv2, "face"):
            raise RuntimeError("opencv-contrib-python is required for cv2.face.")
        emotion_path = os.path.join(ROOT, "models", "emotion_model.keras")
        identity_path = os.path.join(ROOT, "models", "identity_lbph.yml")
        labels_path = os.path.join(ROOT, "models", "identity_labels.json")
        missing = [p for p in (emotion_path, identity_path, labels_path) if not os.path.isfile(p)]
        if missing:
            raise RuntimeError("Missing trained model files. Train identity and emotion models first.")

        import tensorflow as tf

        self.emotion_model = tf.keras.models.load_model(emotion_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(identity_path)
        with open(labels_path, "r", encoding="utf-8") as handle:
            self.names = json.load(handle)

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
            return
        if self.registration:
            messagebox.showinfo("Registration active", "Stop registration before starting recognition.")
            return
        try:
            self.load_models()
        except Exception as exc:
            messagebox.showerror("Models unavailable", str(exc))
            self.log_message("Error: " + str(exc))
            return

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            messagebox.showerror("Camera unavailable", "Cannot open the webcam.")
            return
        self.running = True
        self.start_button.configure(text="Stop Recognition")
        self.set_camera_status(True, "Camera running")
        self.log_message("Live recognition started.")
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.start_button.configure(text="Start Recognition")
        self.set_camera_status(False, "Camera stopped")
        self.log_message("Camera stopped.")

    def toggle_registration(self):
        if self.registration:
            self.stop_registration()
            return
        name = self.name_entry.get().strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9 _-]+", name):
            messagebox.showerror("Invalid name", "Use letters, numbers, spaces, _ or -.")
            return

        folder = os.path.join(ROOT, "data", "identity", name.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)
        self.registration_name = name
        self.registration_folder = folder
        self.capture_count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if self.capture_count >= 60:
            messagebox.showinfo("Registration complete", "This participant already has 60 captures.")
            return
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            messagebox.showerror("Camera unavailable", "Cannot open the webcam.")
            return
        self.registration = True
        self.running = True
        self.register_button.configure(text="Stop Registration")
        self.set_camera_status(True, "Capturing faces")
        self.log_message("Registration started for " + name + ".")
        self.update_frame()

    def stop_registration(self):
        self.registration = False
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.register_button.configure(text="Start 60 Captures")
        self.set_camera_status(False, "Camera stopped")
        self.log_message("Registration stopped. Saved " + str(self.capture_count) + " captures.")

    def update_frame(self):
        if not self.running or self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.stop_camera()
            return

        if self.registration:
            frame = self.draw_registration_frame(frame)
        else:
            frame = self.draw_recognition_frame(frame)
        self.show_frame(frame)
        self.root.after(30, self.update_frame)

    def draw_registration_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces) > 0 and self.capture_count < 60:
            x, y, w, h = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)[0]
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            filename = os.path.join(self.registration_folder, f"{self.capture_count:03}.jpg")
            cv2.imwrite(filename, face)
            self.capture_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (34, 197, 94), 2)
            cv2.putText(
                frame, f"Saved {self.capture_count}/60", (x, max(25, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 197, 94), 2
            )
            if self.capture_count >= 60:
                self.log_message("Captured 60 images for " + self.registration_name + ".")
                self.root.after(10, self.stop_registration)
        return frame

    def draw_recognition_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x, y, w, h in self.cascade.detectMultiScale(gray, 1.2, 5, minSize=(70, 70)):
            roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, distance = self.recognizer.predict(roi)
            identity = self.names.get(str(label), "Unknown") if distance < 75 else "Unknown person"
            color = (34, 197, 94) if identity != "Unknown person" else (239, 68, 68)

            emotion_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            emotion_input = emotion_roi.astype("float32").reshape(1, 48, 48, 1) / 255.0
            probabilities = self.emotion_model.predict(emotion_input, verbose=0)[0]
            expression = EMOTIONS[int(np.argmax(probabilities))]

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame, identity, (x, max(24, y-28)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )
            cv2.putText(frame, "Expression: " + expression, (x, y+h+24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 204, 21), 2)
        return frame

    def show_frame(self, frame):
        height, width = frame.shape[:2]
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        scale = min(canvas_width / width, canvas_height / height)
        new_width, new_height = max(1, int(width * scale)), max(1, int(height * scale))
        resized = cv2.resize(frame, (new_width, new_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("placeholder")
        self.canvas.delete("frame")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2, image=self.photo, tags="frame"
        )

    def open_image(self):
        if self.running:
            if self.registration:
                self.stop_registration()
            else:
                self.stop_camera()
        path = filedialog.askopenfilename(
            title="Choose a face image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.load_models()
        except Exception as exc:
            messagebox.showerror("Models unavailable", str(exc))
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Invalid image", "Could not read the selected image.")
            return
        self.show_frame(self.draw_recognition_frame(frame))
        self.log_message("Analyzed image: " + os.path.basename(path))

    def run_script(self, command, title):
        self.log_message("Starting " + title + "...")

        def worker():
            try:
                process = subprocess.Popen(
                    command, cwd=ROOT,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                for line in process.stdout:
                    self.root.after(0, self.log_message, line.rstrip())
                code = process.wait()
                message = title + " completed." if code == 0 else title + " failed with exit code " + str(code)
                self.root.after(0, self.log_message, message)
                if code == 0:
                    self.root.after(0, messagebox.showinfo, "Complete", message)
                else:
                    self.root.after(0, messagebox.showerror, "Failed", message)
            except Exception as exc:
                self.root.after(0, self.log_message, "Error: " + str(exc))
                self.root.after(0, messagebox.showerror, "Failed", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def train_identity(self):
        identity_root = os.path.join(ROOT, "data", "identity")
        os.makedirs(identity_root, exist_ok=True)
        if not any(os.path.isdir(os.path.join(identity_root, item))
                   for item in os.listdir(identity_root)):
            messagebox.showwarning("No samples", "Register at least one participant first.")
            return
        self.run_script([sys.executable, "train_identity.py"], "Identity model training")

    def train_emotion(self):
        csv_path = os.path.join(ROOT, "data", "fer2013.csv")
        if not os.path.isfile(csv_path):
            messagebox.showerror("Dataset missing", "Place fer2013.csv in the data folder first.")
            return
        try:
            epochs = max(1, int(self.epochs.get()))
        except ValueError:
            epochs = 25
        self.run_script(
            [sys.executable, "train_emotion.py", "--csv", csv_path,
             "--epochs", str(epochs), "--batch-size", "64"],
            "Emotion model training"
        )

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    FaceRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
