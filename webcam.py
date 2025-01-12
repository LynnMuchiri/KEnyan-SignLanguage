import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)  # Use webcam (ID 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw bounding boxes
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, confidence, class_id = box
        label = f"Gesture {int(class_id)}: {confidence:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Show the frame
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
labels={
    0:"church",
    1:"coffee",
    2:"friend",
    3:"good",
    4:"love",
    5:"me",
    6:"mo",
    7: "mosque",
    8: "okay",
    9: "open",
    10: "satisfied",
    11: "seat",
    12: "spoon",
    13: "tea",
    14: "temple",
    15: "you",
}