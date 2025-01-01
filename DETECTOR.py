from cgitb import text
from distutils import command

# from msilib.schema import ListBox
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import time

import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
emotion_model_path = "./models/emotion_model.hdf5"
emotion_labels = get_labels("fer2013")

# hyper-parameters for bounding boxes shape
window_frame = 10
emotion_offsets = (20, 40)

# loading model
face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
# emotion_classifier = load_model(emotion_model_path)

###
from tensorflow.keras.optimizers import Adam

emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_classifier.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


###

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# Proceso para el contador
proceso = 0

# Guardado de datos
f = open("SaveEmotions.csv", "w+")


# Metodo para elegir video o camara
def video_de_entrada():
    global cap
    if selected.get() == 1:
        path_video = filedialog.askopenfilename(
            filetypes=[
                ("all video format", ".mp4"),
                ("all video format", ".avi"),
                ("all video format", ".mov"),
                ("all video format", ".wmv"),
                ("all video format", ".mkv"),
                ("all video format", ".flv"),
                ("all video format", ".f4v"),
                ("all video format", ".swf"),
                ("all video format", ".webm"),
                ("all video format", ".m4v"),
                ("all video format", ".html5"),
                ("all video format", ".avchd"),
            ]
        )
        if len(path_video) > 0:
            btnEnd.configure(state="active")
            btnEnd.configure(bg="#778899")
            rad1.configure(state="disabled")
            # rad2.configure(state="disabled")
            pathInputVideo = "..." + path_video[-20:]
            lblInfoVideoPath.configure(text=pathInputVideo)
            cap = cv2.VideoCapture(path_video)
            visualizar()

    if selected.get() == 2:
        btnEnd.configure(state="active")
        btnEnd.configure(bg="#778899")
        rad1.configure(state="disabled")
        # rad2.configure(state="disabled")
        lblInfoVideoPath.configure(text="")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        visualizar()


# Metodo para visualizar el video
def visualizar():
    global cap
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = deteccion_facilal(frame)
        frame = imutils.resize(frame, width=1154, height=780)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)
        iniciar()
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, visualizar)
    else:
        lblVideo.image = ""
        lblInfoVideoPath.configure(text="")
        rad1.configure(state="active")
        # rad2.configure(state="active")
        selected.set(0)
        btnEnd.configure(state="disabled")
        # f.close()
        parar()
        cap.release()


# Metodo para detectar la expresion
def deteccion_facilal(frame):
    # Counters
    AngerCounter = 0
    SadCounter = 0
    HappyCounter = 0
    SurpriceCounter = 0

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > window_frame:
            emotion_window.pop(0)

        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == "angry":
            color = emotion_probability * np.asarray((255, 0, 0))
            AngerCounter += 1
            # print("Times Anger detected: ",  AngerCounter)
            f.write(" Numero de eventos Enojo: %d\r\n " % AngerCounter)
            listBox.insert(
                0, " Numero de eventos Enojo: %d\r\n " % AngerCounter, proceso
            )

        elif emotion_text == "sad":
            color = emotion_probability * np.asarray((0, 0, 255))
            SadCounter += 1
            # print("Times Sad detected", SadCounter)
            f.write(" Numero de eventos Triste: %d\r\n " % AngerCounter)
            listBox.insert(
                0, " Numero de eventos Triste: %d\r\n " % AngerCounter, proceso
            )

        elif emotion_text == "happy":
            color = emotion_probability * np.asarray((255, 255, 0))
            HappyCounter += 1
            # print("Times Happiness detected",HappyCounter)
            f.write(" Numero de eventos Felicidad: %d\r\n " % AngerCounter)
            listBox.insert(
                0, " Numero de eventos Felicidad: %d\r\n " % AngerCounter, proceso
            )

        elif emotion_text == "surprise":
            color = emotion_probability * np.asarray((0, 255, 255))
            SurpriceCounter += 1
            # print("Times surprice detected", SurpriceCounter)
            f.write(" Numero de eventos Sorpresa: %d\r\n " % AngerCounter)
            listBox.insert(
                0, " Numero de eventos Sorpresa: %d\r\n " % AngerCounter, proceso
            )
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        frame = bgr_image

    return frame


# Método para iniciar el contador
def iniciar(h=0, m=0, s=0):
    global proceso
    # Verificamos si los segundos y los minutos son mayores a 60
    # Verificamos si las horas son mayores a 24
    if s >= 60:
        s = 0
        m = m + 1
        if m >= 60:
            m = 0
            h = h + 1
            if h >= 24:
                h = 0
    # etiqueta que muestra el cronometro en pantalla
    time["text"] = str(h) + ":" + str(m) + ":" + str(s)
    # iniciamos la cuenta progresiva de los segundos
    proceso = time.after(1000, iniciar, (h), (m), (s + 1))


# Método para limpiar frame del video
def finalizar_limpiar():
    rad1.configure(bg="#008B8B")
    # rad2.configure(bg='#008B8B')
    # f.close()
    parar()
    lblVideo.image = ""
    lblInfoVideoPath.configure(text="")
    rad1.configure(state="active")
    # rad2.configure(state="active")
    selected.set(0)
    cap.release()


# Método para detener el contador
def parar():
    global proceso
    time.after_cancel(proceso)


# Creación de la GUI ventana
cap = None
root = Tk()
root.title("Detector")
root.configure(bg="#008B8B")
root.columnconfigure([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], minsize=1, weight=1)

lblInfo1 = Label(root, text="Facial Expression Detection", font="bold")
lblInfo1.grid(column=0, row=0, columnspan=2)
lblInfo1.configure(bg="#008B8B")

# Creación de RadioButton para seleccionar video
selected = IntVar()
rad1 = Radiobutton(
    root,
    text="Upload video ",
    width=20,
    value=1,
    variable=selected,
    command=video_de_entrada,
    font="italic",
)
# rad2 = Radiobutton(root, text="Streaming", width=20, value=2, variable=selected, command=video_de_entrada, font="italic")
rad1.grid(column=0, row=1, columnspan=2)
# rad2.grid(column=1, row=1)
rad1.configure(bg="#008B8B")
# rad2.configure(bg='#008B8B')

lblInfoVideoPath = Label(root, text="", width=20)
lblInfoVideoPath.grid(column=0, row=2)
lblInfoVideoPath.configure(bg="#008B8B")

# Creación de Label donde se visualizara el video
lblVideo = Label(root)
lblVideo.grid(column=0, row=3, columnspan=2)
lblVideo.configure(bg="#000000")

# Creación de Botón para detener el video y limpiar el frame
btnEnd = Button(
    root,
    text="Finish / Clean ",
    state="disabled",
    command=finalizar_limpiar,
    font="italic",
)
btnEnd.grid(column=0, row=4, columnspan=2, pady=10)
btnEnd.configure(bg="#778899")

# Creación de ListBox para visualizar los eventos detectados
listBox = Listbox(root, font=("italic", "12"))
listBox.insert(0, "")
listBox.place(width=290, height=650, x=1215, y=86)
listBox.configure(bg="#778899")

# Creación Label del contador
time = Label(root, fg="black", width=20, font=("", "18"))
time.place(x=1210, y=37)
time.configure(bg="#008B8B")

root.mainloop()
