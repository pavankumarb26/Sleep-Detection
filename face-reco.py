import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("Failed to load Haar Cascade")
    exit()

webcam = cv2.VideoCapture(0)

while True:
    ret, img = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

webcam.release()
cv2.destroyAllWindows()