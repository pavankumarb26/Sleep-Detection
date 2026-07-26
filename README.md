# Real-time Sleep Detection Alarm System

A real-time Computer Vision application built in Python that uses your laptop's webcam to monitor eye state and sound an alarm if you begin to fall asleep while working.

---

## 🛠️ Project Setup & Installation

### 1. Install Dependencies
Make sure you have Python installed (the project is fully compatible with Python 3.12 and Python 3.13). Run the following command in your terminal to install the necessary libraries:

```bash
pip install -r requirements.txt
```

### 2. Generate the Alarm Sound
Since standard Python cannot write native MP3 files programmatically without external dependencies like FFmpeg, we generate a synthetic alarm sound in `.wav` format. Run the generator script to create the sound file:

```bash
python generate_alarm.py
```
This creates a 1.5-second high-pitched beep named `alarm.wav` in the project folder.

### 3. Run the Application
Start the real-time sleep detection system:

```bash
python main.py
```
* Press **`ESC`** or **`q`** inside the camera window to exit and release resources.

---

## 📂 Project Structure

* **[main.py](file:///c:/Users/pavan/OneDrive/Desktop/Face-detection/main.py)**: The main application containing the video frame capture, face landmark tracking, Eye Aspect Ratio (EAR) calculations, drowsiness state machine, and audio playback loop.
* **[requirements.txt](file:///c:/Users/pavan/OneDrive/Desktop/Face-detection/requirements.txt)**: Specifies stable versions of libraries (`opencv-contrib-python<5`, `mediapipe`, `numpy`, and `pygame`).
* **[generate_alarm.py](file:///c:/Users/pavan/OneDrive/Desktop/Face-detection/generate_alarm.py)**: Helper script using Python's standard `wave` and `struct` libraries to write a synthetic sine-wave beep sound.
* **`alarm.wav`**: The generated alarm file played when the user falls asleep (you can replace this with your own WAV or MP3 file, just make sure to rename it to `alarm.wav` or update the filename in `main.py`).
* **`face_landmarker.task`**: The official pre-trained Google MediaPipe Face Landmarker model asset used for landmark coordinate extraction.

---

## 🧠 Core Computer Vision Concepts Explained

### 1. How OpenCV Captures Frames
OpenCV accesses the camera using the driver pipeline via **`cv2.VideoCapture(0)`** (where `0` represents your default built-in camera).
* **Frame Loop:** Inside the loop, **`cap.read()`** grabs the latest image frame from your camera driver's hardware buffer. It returns:
  1. A status boolean (`ret`) which is `True` if a frame was successfully read.
  2. The frame itself (`frame`) as a 3D NumPy array of size `(Height, Width, 3)` containing color channels in **BGR** (Blue, Green, Red) format.
* **Refreshes & Inputs:** **`cv2.waitKey(1)`** is critical. It halts execution for 1ms to allow your operating system's window manager to repaint the window. Without it, the camera feed would freeze immediately.

### 2. How MediaPipe Tracks Landmarks
MediaPipe uses a two-stage deep-learning pipeline:
1. **Face Detection (BlazeFace):** A lightweight detector scans the frame to crop a bounding box containing the face.
2. **Face Mesh Prediction:** A convolutional neural network tracks **478 3D landmarks** within the cropped face region.
* **BGR to RGB:** MediaPipe models are trained on **RGB** images. OpenCV reads in BGR, so we must run `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` to prevent the network from getting confused by swapped color channels.
* **Normalized Coordinates:** MediaPipe outputs coordinates as fractions from `0.0` to `1.0`. We scale them back to pixel values by multiplying them by the width and height of the camera frame:
  $$\text{Pixel } X = x \cdot \text{Frame Width}$$
  $$\text{Pixel } Y = y \cdot \text{Frame Height}$$

### 3. How Eye Aspect Ratio (EAR) is Calculated
To calculate how open an eye is, we map 6 specific landmark coordinates around each eye.
For an eye represented by points $p_1, p_2, p_3, p_4, p_5, p_6$:
* $p_1$ and $p_4$ are the horizontal corners.
* $p_2, p_3$ are on the upper eyelid, and $p_5, p_6$ are on the lower eyelid.

$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

* **Euclidean Distance:** The distance between points is calculated using the Pythagorean formula $d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$, which we run in Python via `np.linalg.norm()`.
* **Scale Invariance:** Because the vertical eyelid heights are divided by the horizontal width, the value is a **ratio**. If you move closer to the camera, both height and width scale up equally, and the ratio remains constant. This means sitting closer or further from your laptop does not break the sleep detection system!
* **Value Ranges:** 
  * Open eyes: EAR values hover between **`0.25` and `0.32`**.
  * Closed eyes: EAR values drop below **`0.20`**.

### 4. Drowsiness State Machine & Timers
Normal human blinks last **`0.1` to `0.4` seconds**. If we triggered the alarm immediately when the EAR dropped below `0.20`, the alarm would chirp every time you blinked. 
* **The Solution:** We implement a **temporal filter** (a duration check) using Python's `time` library:
  * When your eyes drop below the threshold, we record the start time (`time.time()`).
  * If the eyes remain closed continuously for **1.0 second**, the user state changes to **`Drowsy`**.
  * If they remain closed for **2.5 seconds**, the state changes to **`Sleeping - Wake Up!`** and the alarm sounds.
  * If the eyes open, the start time is cleared and the state immediately returns to **`Awake`**.
* **Why Time-based is better than Frame-based:** If we counted raw frames (e.g. "wait 60 frames"), the delay time would vary depending on your camera's frame rate. In low light, a webcam might drop to 10 FPS, meaning 60 frames would take **6 seconds** instead of 2. Using timestamps ensures the alarm triggers after exactly `2.5` seconds on all machines.
