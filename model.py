from ultralytics import YOLO

# Load the trained model
model_path = 'best.pt'# Update with your model path
model = YOLO(model_path)

# Perform real-time inference using a webcam
print("Starting webcam. Press 'q' to exit.")
model.predict(source=0, show=True)


