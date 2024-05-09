import cv2
import dlib

# Load pre-trained face detection model
detector = dlib.get_frontal_face_detector()

# Load pre-trained face recognition model
# You can replace "model_path" with the path to your pre-trained model file
model_path = "path_to_pretrained_model.dat"
face_recognizer = dlib.face_recognition_model_v1(model_path)

# Load a sample image for testing
image_path = "path_to_sample_image.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Iterate over detected faces
for face in faces:
    # Get facial landmarks
    shape = predictor(gray, face)
    # Compute face descriptor
    face_descriptor = face_recognizer.compute_face_descriptor(gray, shape)
    
    # You can save face descriptors for later use, or compare with other faces
    
    # Draw a rectangle around the face
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
