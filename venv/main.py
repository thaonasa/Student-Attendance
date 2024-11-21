import time
import cv2
import os


face_detector = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_alt.xml')

# def getFaces(image_path):
#     img = cv2.imread(image_path)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
#     count = 0
#     for (x, y, w, h) in faces:
#         img_face = cv2.resize(
#             img[y + 3: y + h - 3, x + 3: x + w - 3], (64, 64))
#         cv2.imwrite(img_path.replace('imgs', 'imgs_face').split(
#             '.jpg')[0] + '_{}.jpg'.format(count), img_face)
#         # cv2.rectangle(img, (x,y), (x + m, y + h), (0, 255, 0), 2)
#         count += 1


# image_path = 'imgs'

# for whatelse in os.listdir(image_path):
#     whatelse_path = os.path.join(image_path, whatelse)
#     for sub_whatelse in os.listdir(whatelse_path):
#         img_path = os.path.join(whatelse_path, sub_whatelse)

#         if not os.path.isdir(whatelse_path.replace('imgs', 'imgs_face')):
#             os.mkdir(whatelse_path.replace('imgs', 'imgs_face'))
#             print(whatelse_path.replace('imgs', 'imgs_face'))

#         if img_path.endswith('.jpg'):
#             getFaces(img_path)

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(
    'Hartlepool factory to manufacture new super fast Covid-19 test machines.mp4')

count = 0
while True:
    OK, frame = cam.read()
    faces = face_detector.detectMultiScale(frame, 1.3, 5)
    time.sleep(0.3)
    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y+2: y+h-2, x+2: x+w-2], (100, 100))
        cv2.imwrite('imgs_roi/roi_{}.jpg'.format(count), roi)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 50), 1)
        count += 1

    cv2.imshow('FRAME', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
# def doSomething():
#     lst = []
#     for i in range(0, 100, 4):
#         lst.append((i, i+1, i+2, i+3))

#     return lst

# funcCall = doSomething()
# for item in funcCall:
#     print(item)
