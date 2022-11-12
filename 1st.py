from deepface import DeepFace
import cv2
img1_path = cv2.imread("Soham0.png")
img2_path = cv2.imread("Soham2.png")

model_name = "Facenet"

verify = DeepFace.verify(img1_path = img1_path, img2_path = img2_path,model_name=model_name)
process = verify["verified"]



if process :
    print("Continue")
else:
    print("Register First")