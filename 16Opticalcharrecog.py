#OPTICAL CHARACTER RECOGNITION
#label reading using OCR->extracting text from image-pytesseract
import cv2
import time
try:
    from PIL import Image
except ImportError:
    import Image
    
import pytesseract
pytesseract.pytesseract.tesseract_cmd= r'C:/Program Files/Tesseract-OCR/tesseract.exe'#r-> bcoz file located in another folder


# Initialize the camera
cam = cv2.VideoCapture(0)
time.sleep(1)# it will pause the program for 1 sec

while True:
    # Capture a frame from the camera
    _,img = cam.read()

    # Display the frame
    print("Press 'esc' to capture a picture")
    cv2.imshow("Camera", img)

    # Check for key press event
    key = cv2.waitKey(1)

    # Check if 'esc' key is pressed to capture the picture
    if key == 27:
        cv2.imwrite("captured.jpg",img)
        print("Picture captured!")
        break

# Release the camera and close windows
cam.release()
cv2.destroyAllWindows()


    
def recText(filename):
    text = pytesseract.image_to_string(Image.open(filename))
    return text

info = recText('captured.jpg')
print(info)
file = open("result.txt","w")
file.write(info)
file.close()
print("written successful")