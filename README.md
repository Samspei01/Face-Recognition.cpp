# FaceLandmarks Project

## Overview
FaceLandmarks is a facial landmark detection project that utilizes OpenCV, TensorFlow Lite, Dlib, and OpenBLAS to detect and process facial features. Additionally, this project includes a script for generating face encodings using `face_recognition`.

## System Requirements
- Raspberry Pi 4 or Raspberry Pi 5
- Raspberry Pi OS (Debian-based Linux)
- At least 4GB of RAM recommended for smooth performance

## Dependencies
To run this project, you need to install the following dependencies:

### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Required Packages
```bash
sudo apt install -y cmake g++ git wget unzip libopencv-dev libopenblas-dev python3-pip python3-opencv
```

### 3. Install OpenCV
```bash
sudo apt install -y libopencv-core-dev libopencv-highgui-dev libopencv-imgproc-dev
```

### 4. Install Dlib
```bash
sudo apt install -y libdlib-dev
```
If you need to build Dlib manually:
```bash
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build && cd build
cmake ..
cmake --build .
sudo make install
cd ../..
```

### 5. Install TensorFlow Lite
```bash
mkdir -p third_party/tflite && cd third_party/tflite
wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.10.0.zip
unzip v2.10.0.zip && mv tensorflow-2.10.0 tensorflow
cd tensorflow
./tensorflow/lite/tools/make/download_dependencies.sh
cd ../..
```

### 6. Install Face Recognition Library
```bash
pip install face-recognition numpy
```

## Building the Project
### 1. Create Build Directory and Compile
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 2. Run the Application
```bash
./demo
```

### 3. Encode Faces for Recognition
To encode a known face, run the following command:
```bash
python3 train_faces.py <known_image_path>
```
This will generate `encoding.json` containing the face encoding.

## Notes
- Ensure your Raspberry Pi has sufficient power supply (5V/3A recommended).
- If you encounter missing libraries, verify their installation with `dpkg -l | grep <library-name>`.
- TensorFlow Lite can be optimized further using Raspberry Pi's acceleration options (e.g., Edge TPU).

## License
This project is licensed under the MIT License.

## Contact
For any issues or contributions, feel free to open an issue or submit a pull request!


