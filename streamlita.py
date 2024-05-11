from easy_facial_recognition import *
import dlib
import cv2
import streamlit as st


def load():
    print('[INFO] Starting System...')
    print('[INFO] Importing pretrained model..')
    pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
    pose_predictor_5_point = dlib.shape_predictor("pretrained_model/shape_predictor_5_face_landmarks.dat")
    face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
    face_detector = dlib.get_frontal_face_detector()
    print('[INFO] Importing pretrained model..')
    print('[INFO] Importing faces...')
    face_to_encode_path = Path('./known_faces')
    files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]

    for file_ in face_to_encode_path.rglob('*.png'):
        files.append(file_)
    if len(files) == 0:
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
    return pose_predictor_68_point, pose_predictor_5_point, face_encoder, face_detector, known_face_names, known_face_encodings


pose_predictor_68_point, pose_predictor_5_point, face_encoder, face_detector, known_face_names, known_face_encodings = load()
st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(frame)
    easy_face_reco(frame, known_face_encodings, known_face_names)
    FRAME_WINDOW.image(frame)

