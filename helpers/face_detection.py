import cv2
import numpy as np
from torchvision import transforms
import torch
 
# might change the way we handle the image based on the dtype of image sent from request
def face_detection(img_path) :
    '''
    Args:
        img_path (string): the path to the saved face(s) image
    Return:
        a torch tensor including the arrays representing all the cropped face from the \
            input image provided by the user            
    '''
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    all_faces = ()
    # # Draw a rectangle around the faces
    # for degree in range (0, 365, 15) :
    #     rotated = ndimage.rotate(gray, degree, reshape=False)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.23,
        minNeighbors=5,
        minSize=(224, 224),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]
        face = img_transform(face)
        all_faces += (face,)

    to_predict = torch.stack(all_faces)

    return to_predict


def img_transform(image) :
    image_trans = np.reshape(image, (image.shape[2], image.shape[0], image.shape[1]))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(
            mean=[0.5686, 0.4505, 0.3990],
            std=[0.2332, 0.2064, 0.1956]
        )
    ])

    image_trans = transform(image)
    return image_trans