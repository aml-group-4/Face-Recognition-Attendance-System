import customtkinter as ctk
import cv2
import mediapipe as mp
import tkinter as tk
import subprocess
from PIL import Image, ImageTk
from customtkinter import FontManager
from scipy.spatial.distance import cosine
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
import GestureRecognition

################### INTERFACE ###################
# BUTTONS
FontManager.load_font("AppleGaramond.ttf")
gr = GestureRecognition.GestureRecognition()
none_spoofed_image = None
# Load the antispoofing model
model = YOLO('antispoof.pt')
# Load the face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
real_face = False
button_width, button_height, font_size, font_size_total = 460, 70, 48, 60
corner_radius, border_color, border_width = 8, "black", 2


def create_button(parent, text, command, fg_color, hover_color, text_color):
    return ctk.CTkButton(parent, text=text, command=command, fg_color=fg_color,
                         text_color=text_color, hover_color=hover_color, font=(
                             "Arial", font_size),
                         width=button_width, height=button_height,
                         corner_radius=corner_radius, border_color=border_color,
                         border_width=border_width)


def register_interface():
    global buttonRegister, buttonCheckIn, buttonBack, register_fill

    # Hide existing buttons
    buttonRegister.pack_forget()
    buttonCheckIn.pack_forget()
    buttonBack.pack_forget()

    # Create and show the fill-in box
    register_fill = ctk.CTkEntry(
        button_frame, width=button_width, height=button_height, corner_radius=corner_radius)
    register_fill.pack(side="left", padx=10)

    buttonRegister = create_button(button_frame, "Register",
                                   lambda: register_user(register_fill.get()),
                                   "#009946", "#007D39", "white")

    buttonRegister.pack(side="left", padx=10)


def main_interface():
    global buttonRegister, buttonCheckIn, buttonBack, real_face

    buttonRegister = create_button(
        button_frame, "Register", register_interface, "#009946", "#007D39", "white")
    buttonRegister.pack(side="left", padx=10)

    buttonCheckIn = create_button(
        button_frame, "Check-in", check_in, "#FF8A00", "#D97500", "white")
    buttonCheckIn.pack(side="left", padx=10)

    buttonBack = create_button(
        button_frame, "Back", back, "#F22C17", "#BF1200", "white")
    buttonBack.pack(side="left", padx=10)

    if real_face == False:
        buttonRegister.configure(state="disabled")
        buttonCheckIn.configure(state="disabled")
    else:
        buttonRegister.configure(state="normal")
        buttonCheckIn.configure(state="normal")


def update_total_employees_label():
    total_employees_label.configure(
        text=f"Total Employees: {len(checked_in_employees)}")



def back():
    subprocess.Popen(["python", "app.py"])
    app.quit()

################### VIDEO ###################

# ANTI-SPOOFING


def update_frame():
    global real_face
    ret, frame = cap.read()
    if ret:
        # Process frame with MediaPipe or other processing here
        image = gr.load_image_from_file(frame)

        gesture_detected = gr.recognize_hand_gesture(image)
        print(str(gr.gesture_challenge_list) + " | " +
              str(gr.skip_gesture_list) + " | " + str(gesture_detected))

        gr.update_challenge_state(gesture_detected)

        formatted_gesture_challenge = ', '.join(
            [gesture.replace('_', ' ') for gesture in gr.gesture_challenge_list])

        challenge_label.configure(
            text="Challenge: " + formatted_gesture_challenge
        )

        if gr.check_challenge_complete():
            gr.verified_image = frame
            gr.reset_challenge()

            # update verified label
            liveness_label.configure(
                text="Liveness: True", text_color="#009946")

            antispoof_label.configure(
                text="Checking Spoof Face ...", text_color="#FF0000")

            # check whether the face is spoof
            antispoof_results = model(
                preprocess_image_antispoof(frame))
            # View results
            thres = 0.7
            for r in antispoof_results:
                if (r.probs.data[0] > thres):
                    antispoof_label.configure(
                        text="Antispoof: True", text_color="#009946")
                    real_face = True

                else:
                    antispoof_label.configure(
                        text="Antispoof: False", text_color="#FF0000")
                    real_face = False

        if real_face == False:
            buttonRegister.configure(state="disabled")
            buttonCheckIn.configure(state="disabled")
        else:
            buttonRegister.configure(state="normal")
            buttonCheckIn.configure(state="normal")

        # Convert the frame to PhotoImage
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, update_frame)  # Continue updating the frame


################### UNSUPERVISED MODEL ###################
RECOGNITION_THRESHOLD = 0.015

# Load the FaceNet model
def load_facenet_model():
    facenet = FaceNet()
    model = facenet.model  # Access the Keras model in FaceNet
    return model

embedding_model = load_facenet_model()
embedding_model.load_weights('v4_facenet_siamese_network_embedding.h5')

user_embeddings = {}


def preprocess_image(image):
    image = cv2.resize(image, (160, 160))  # Resize image to 160x160 for FaceNet
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return np.expand_dims(image, axis=0)


def generate_embedding(image):
    preprocessed_image = preprocess_image(image)
    return embedding_model.predict(preprocessed_image)[0]


def crop_face(image):
    margin_x = 30
    margin_y = 50

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(face) == 0:
        for (x, y, w, h) in face:

            # crop the detected face region
            face_img = image[y - margin_y:y + h + margin_y,
                             x - margin_x:x + w + margin_x]

            return face_img
    else:
        return None


def preprocess_image_antispoof(image):
    face_img = crop_face(image)
    if (face_img is None):
        return image
    # face_img = cv2.resize(face_img, (640, 640))  # Resize image
    return face_img

def register_user(user_id):
    global status_label, buttonRegister, register_fill, real_face

    try:
        # ret, frame = cap.read()
        # if not ret:
        #     status_label.configure(text="Error capturing frame from camera.")
        #     return

        frame = gr.access_verified_image()
        real_face = False

        # if (frame is None):
        #     status_label.configure(text="Please complete the challenge first.")
        #     return

        # # check whether the face is spoof
        # antispoof_results = model(preprocess_image_antispoof(frame))

        # # View results
        # thres = 0.7
        # for r in antispoof_results:
        #     if (r.probs.data[0] > thres):
        #         antispoof_label.configure(
        #             text="Liveness: True", text_color="#009946")
        #     else:
        #         antispoof_label.configure(
        #             text="Liveness: False", text_color="#FF0000")
        #         return

        embedding = generate_embedding(frame)
        user_embeddings[user_id] = embedding
        status_label.configure(text=f"User {user_id} registered successfully.")

        liveness_label.configure(
            text="Liveness: False", text_color="#FF0000")
        antispoof_label.configure(
            text="Antispoof: False", text_color="#FF0000")
        register_fill.pack_forget()
        buttonRegister.pack_forget()

        main_interface()

    except Exception as e:
        status_label.configure(text=f"Error during registration: {str(e)}")
        print(f"Error during registration: {str(e)}")

def update_check_in_employees_label():
    # Join all checked-in employee names with a newline character
    employees_list_str = '\n'.join(checked_in_employees)
    check_in_employees_label.configure(text=f"Check-In Employees:\n{employees_list_str}")

def check_in():
    global status_label, checked_in_employees, real_face

    try:
        # ret, frame = cap.read()
        # if not ret:
        #     status_label.configure(text="Error capturing frame from camera.")
        #     return

        frame = gr.access_verified_image()
        real_face = False

        # if (frame is None):
        #     status_label.configure(text="Please complete the challenge first.")
        #     return

        # # check whether the face is spoof
        # antispoof_results = model(preprocess_image_antispoof(frame))

        # # View results
        # thres = 0.7
        # for r in antispoof_results:
        #     if (r.probs.data[0] > thres):
        #         antispoof_label.configure(
        #             text="Liveness: True", text_color="#009946")
        #     else:
        #         antispoof_label.configure(
        #             text="Liveness: False", text_color="#FF0000")
        #         return

        new_embedding = generate_embedding(frame)
        min_distance = float('inf')
        recognized_user_id = "Unknown"

        for user_id, embedding in user_embeddings.items():
            distance = cosine(new_embedding, embedding)
            if distance < min_distance:
                min_distance = distance
                recognized_user_id = user_id
            
        # Update status label based on recognition result
        if min_distance <= RECOGNITION_THRESHOLD:
            checked_in_employees.add(recognized_user_id)
            update_total_employees_label()
            update_check_in_employees_label()  # Update the check-in employees label
            status_label.configure(
                text=f"Recognized User: {recognized_user_id}")
        else:
            status_label.configure(text="User not recognized.")

        liveness_label.configure(
            text="Liveness: False", text_color="#FF0000")

        antispoof_label.configure(
            text="Antispoof: False", text_color="#FF0000")

    except Exception as e:
        status_label.configure(text=f"Error during recognition: {str(e)}")


################### APP IMPLEMENTATION ###################
app = ctk.CTk()
app.title("Face Recognition Attendance System | Unsupervised model by Iman")
app.geometry("1440x850")
ctk.set_appearance_mode("light")

# VIDEO
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)
main_frame = ctk.CTkFrame(app, fg_color='#FFF8E9')
main_frame.pack(expand=True, fill='both')
video_label = tk.Label(main_frame, width=930, height=650)
video_label.grid(row=0, column=0, padx=10, rowspan=12, columnspan=2)


# TOTAL EMPLOYEES
checked_in_employees = set()

# TOTAL EMPLOYEES
total_employees_label = ctk.CTkLabel(
    main_frame, text=f"Total Employees: {len(checked_in_employees)}", font=("Apple Garamond", font_size_total))
total_employees_label.grid(row=0, column=2, sticky="nw", padx=10)

# CHECK-IN EMPLOYEES
check_in_employees_label = ctk.CTkLabel(
    main_frame, text=f"Check-In Employees: ", font=("Arial", 40))
check_in_employees_label.grid(row=1, column=2, sticky="nw", padx=10, pady=10)

liveness_label = ctk.CTkLabel(
    main_frame, text="Liveness: False", font=("Apple Garamond", font_size_total), text_color="#FF0000")
liveness_label.grid(row=10, column=2, sticky="sw", padx=10)

antispoof_label = ctk.CTkLabel(
    main_frame, text="AntiSpoofed: False", font=("Apple Garamond", font_size_total), text_color="#FF0000")
antispoof_label.grid(row=11, column=2, sticky="sw", padx=10)


# Format the gesture challenge list
formatted_gesture_challenge = ', '.join([gesture.replace('_', ' ') for gesture in gr.gesture_challenge_list])

# Create the label with the formatted text
challenge_label = ctk.CTkLabel(
    main_frame, text="Challenge: " + formatted_gesture_challenge, font=("Apple Garamond", 30))
challenge_label.grid(row=13, column=1, sticky="sw", padx=10)

# STATUS
status_label = ctk.CTkLabel(main_frame, text="", font=("Arial", font_size))
# Adjust grid positioning as needed
status_label.grid(row=14, column=0, padx=10, columnspan=2)


# BUTTON FRAME
button_frame = ctk.CTkFrame(main_frame, fg_color='#FFF8E9')
button_frame.grid(row=15, column=0, columnspan=3, pady=20)
main_interface()


################### LOOP ###################
update_frame()
app.mainloop()
cap.release()