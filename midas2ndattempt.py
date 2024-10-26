import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.spatial import distance

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load MiDaS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

# Define image transformation pipeline for MiDaS
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def calculate_3d_distance(landmark1, landmark2):
    """Calculates the 3D Euclidean distance between two landmarks."""
    point1 = np.array([landmark1.x, landmark1.y, landmark1.z])
    point2 = np.array([landmark2.x, landmark2.y, landmark2.z])
    return distance.euclidean(point1, point2)

def normalize_z(landmarks, depth_map, image_shape):
    """Normalizes z-values of landmarks using depth information from MiDaS."""
    h, w = image_shape[:2]

    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        # Clamp coordinates to be within image bounds
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)

        # Use depth map to adjust z value
        landmark.z = depth_map[y, x]  # Set z as the depth value from MiDaS

    return landmarks

def check_golden_ratio(measured_ratio, ideal_ratio=1.618, tolerance=0.05):
    """Checks if the measured ratio is within the acceptable range of the golden ratio."""
    return ideal_ratio - tolerance <= measured_ratio <= ideal_ratio + tolerance

def estimate_depth(image):
    """Estimates depth map from MiDaS model."""
    # Convert image to RGB and PIL format
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    input_tensor = transform(img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Get depth map from MiDaS
    with torch.no_grad():
        depth = midas(input_tensor).squeeze().cpu().numpy()

    # Resize depth map to original image size
    depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
    return depth

def detect_face_defects(image):
    """Detects defects in a human face based on golden ratio proportions using MiDaS for depth."""
    # Convert image to RGB as MediaPipe works with RGB images
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate depth map using MiDaS
    depth_map = estimate_depth(image_rgb)

    # Process the image to get facial landmarks
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks on the image (optional)
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Normalize landmarks z-values using depth map
            normalized_landmarks = normalize_z(face_landmarks.landmark, depth_map, image.shape)

            # Calculate face width and length
            face_width = calculate_3d_distance(normalized_landmarks[33], normalized_landmarks[263])  # Eye corners
            face_length = calculate_3d_distance(normalized_landmarks[10], normalized_landmarks[199])  # Top of the forehead to chin
            print(face_length)
            print(face_width)
            length_to_width_ratio = face_length / face_width
            print(f"Calculated Ratio: {length_to_width_ratio}")
            if not check_golden_ratio(length_to_width_ratio):
                print("Defect detected: Face length to width ratio is off.")
            else:
                print("Face proportions are within the golden ratio.")
    else:
        print("No face detected or landmarks could not be extracted.")

def main():
    # Load image from file
    image_path = "frontface2.jpeg    " #ce with your image path
    image = cv2.imread(image_path)

    #if image is not None:
    detect_face_defects(image)

        # Show the image with drawn landmarks
    cv2.imshow('Face Analysis', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  #  else:
    print("Image not found or unable to load.")

if __name__ == "__main__":
    main()
