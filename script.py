import cv2
import numpy as np
import os

# Define paths
data_path = "dataset"
if not os.path.exists(data_path):
    os.makedirs(data_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Capture face images
def capture_faces(name):
    cap = cv2.VideoCapture(0)
    count = 0
    user_path = os.path.join(data_path, name)
    if not os.path.exists(user_path):
        os.makedirs(user_path)
    
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            cv2.imwrite(f"{user_path}/{count}.jpg", face)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Face Capture", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Train the model
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    names = {}
    label_id = 0
    
    for name in os.listdir(data_path):
        user_path = os.path.join(data_path, name)
        if not os.path.isdir(user_path):
            continue
        
        names[label_id] = name
        
        for file in os.listdir(user_path):
            img_path = os.path.join(user_path, file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(label_id)
        
        label_id += 1
    
    recognizer.train(faces, np.array(labels))
    recognizer.save("face_recognizer.yml")
    np.save("names.npy", names)
    print("Training complete!")

# Recognize faces
def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_recognizer.yml")
    names = np.load("names.npy", allow_pickle=True).item()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            label, confidence = recognizer.predict(face)
            name = names.get(label, "Unknown")
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        cv2.imshow("Face Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1: Capture Faces")
    print("2: Train Model")
    print("3: Recognize Faces")
    choice = input("Enter your choice: ")
    if choice == "1":
        name = input("Enter name: ")
        capture_faces(name)
    elif choice == "2":
        train_model()
    elif choice == "3":
        recognize_faces()
