import cv2

cap = cv2.VideoCapture(0)  # Open default camera (0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
