# Code Anis - Defend Intelligence
from email.mime.application import MIMEApplication
from os.path import basename

import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import argparse
from pathlib import Path
import os
import ntpath
import time
#packege pour email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ntpath

names =[]
parser = argparse.ArgumentParser(description='Easy Facial Recognition App')
parser.add_argument('-i', '--input', type=str, required=True, help='directory of input known faces')

print('[INFO] Starting System...')
print('[INFO] Importing pretrained model..')
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("pretrained_model/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
print('[INFO] Importing pretrained model..')



def transform(image, face_locations):# 1-fontion qui reteurn une dimention de face qans l image
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left() # determiner les dementions de visage dans une vecteur
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0) #
        coord_faces.append(coord_face) # ajoute les dementions de visage dans une liste coord_faces mais conserve la forme d'itérable
    return coord_faces


def encode_face(image):
    face_locations = face_detector(image, 1)  #face_detector():fonction qui detecter une visage dans l image
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # DETECT FACES
        shape = pose_predictor_68_point(image, face_location) # Determiné (the facial landmarks) les 68 points dans le visage qui deja detecté
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))#-----------------------------------------
        # GET LANDMARKS
        shape = face_utils.shape_to_np(shape) #puis convertissez les coordonnées du point de repère (x, y) en un tableau NumPy
        landmarks_list.append(shape)# ajoute les coordonnées dans une liste
    face_locations = transform(image, face_locations) #1
    return face_encodings_list, face_locations, landmarks_list


def easy_face_reco(frame, known_face_encodings, known_face_names):
    rgb_small_frame = frame[:, :, ::-1]
    # ENCODING FACE
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # CHECK DISTANCE BETWEEN KNOWN FACES AND FACES DETECTED
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Alert!!"
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        if name == "Alert!!":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)# used to draw a rectangle on any image.
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0,0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)#used to draw a text string on any image
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0,255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            seconds = time.time()
            local_time = time.ctime(seconds)

            if len(names) > 0:
                if name not in names[:]:
                    print(names[:])
                    names.append(name)
            if len(names) == 0:
                names.append(name)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1) # used to draw a circle on any image

def senemial ():
    email = 'zeko1616naim@gmail.com'
    password = 'zakaria1212'
    send_to_email = 'zakarianaim56@gmail.com'
    subject = 'detection'
    message = 'detection'
    filename = 'students.txt'
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = send_to_email
    msg['Subject'] = subject
    body = message
    with open(filename,'r') as f:
        attachment = MIMEApplication( f.read(), Name= basename(filename))
        attachment['Content-Disposition'] = 'attachment; filename="{}"'.format(basename(filename))
    msg.attach(attachment)
    #filename = ntpath.basename(file_location)
    #attachment = open(file_location, "r")
    #part = MIMEBase('application', 'octet-stream')
    #part.set_payload((attachment).read())
    #encoders.encode_base64(part)
    #part.add_header('Content-Disposition', "attachment; filename=%s" % filename)
    #msg.attach(part)
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.starttls()
    server.login(email, password)
    text = msg.as_string()
    server.sendmail(email, send_to_email, text)
    server.quit()


if __name__ == '__main__':
    args = parser.parse_args()

    print('[INFO] Importing faces...')
    face_to_encode_path = Path(args.input)
    files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]

    for file_ in face_to_encode_path.rglob('*.png'):
        files.append(file_)
    if len(files)==0:
        raise ValueError('No faces detect in the directory: {}'.format(face_to_encode_path))
    known_face_names = [os.path.splitext(ntpath.basename(file_))[0] for file_ in files]

    known_face_encodings = []
    for file_ in files:
        image = PIL.Image.open(file_)
        print(file_)
        image = np.array(image)
        face_encoded = encode_face(image)[0][0]
        known_face_encodings.append(face_encoded)

    print('[INFO] Faces well imported')
    print('[INFO] Starting Webcam...')
########## camera
    out = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*"MJPG"), 2, (512, 256))
    video_capture = cv2.VideoCapture(0)
    listeMotsFichier1 = []
    motCourant = None
    print('[INFO] Webcam well started')
    print('[INFO] Detecting...')
    while True:
        ret, frame = video_capture.read()
        easy_face_reco(frame, known_face_encodings, known_face_names)
        cv2.imshow('Easy Facial Recognition App', frame)
        out.write(cv2.resize(frame, (512, 256)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fichier1 = open('students.txt','r')
        motCourant = fichier1.readline()


        with open(r"students.txt", 'w') as fp:
            for item in names:
                fp.write("%s\n" % item)
    
    print('[INFO] Stopping System')
    out.release()
    video_capture.release()
    cv2.destroyAllWindows()
