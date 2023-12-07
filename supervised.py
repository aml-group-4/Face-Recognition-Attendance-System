import customtkinter as ctk
import cv2
import mediapipe as mp
import tkinter as tk
import subprocess
from PIL import Image, ImageTk
from customtkinter import FontManager
from scipy.spatial.distance import cosine
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import GestureRecognition

################### INTERFACE ###################
# BUTTONS
FontManager.load_font("AppleGaramond.ttf")
gr = GestureRecognition.GestureRecognition()
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


def update_total_students_label():
    total_students_label.configure(
        text=f"Total Students: {len(checked_in_students)}")


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


################### SUPERVISED MODEL ###################
RECOGNITION_THRESHOLD = 0.3

embedding_model = tf.keras.models.load_model('supervised_embedding.h5')

user_embeddings = {}


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


def preprocess_image(image):
    image = cv2.resize(image, (375, 375))  # Resize image
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)


def generate_embedding(image):
    preprocessed_image = preprocess_image(image)
    return embedding_model.predict(preprocessed_image)[0]


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


def update_check_in_students_label():
    # Join all checked-in student names with a newline character
    students_list_str = '\n'.join(checked_in_students)
    check_in_students_label.configure(
        text=f"Check-In Students:\n{students_list_str}")


def check_in():
    global status_label, checked_in_students, real_face

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
            checked_in_students.add(recognized_user_id)
            update_total_students_label()
            update_check_in_students_label()  # Update the check-in students label
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
app.title("Face Recognition Attendance System")
app.geometry("1440x850")
ctk.set_appearance_mode("light")

# VIDEO
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)
main_frame = ctk.CTkFrame(app, fg_color='#FFF8E9')
main_frame.pack(expand=True, fill='both')
video_label = tk.Label(main_frame, width=930, height=650)
video_label.grid(row=0, column=0, padx=10, rowspan=3, columnspan=2)


# TOTAL STUDENT
checked_in_students = set()

total_students_label = ctk.CTkLabel(
    main_frame, text=f"Total Students: {len(checked_in_students)}", font=("Apple Garamond", font_size_total))
# Positioned beside the video label
total_students_label.grid(row=0, column=2, sticky="nw", padx=10)


check_in_students_label = ctk.CTkLabel(
    main_frame, text=f"Check-In Students: ", font=("Arial", 40))
# Positioned beside the video label
check_in_students_label.grid(row=1, column=2, sticky="nw", padx=10)

antispoof_label = ctk.CTkLabel(
    main_frame, text="Antispoof: False", font=("Apple Garamond", font_size_total), text_color="#FF0000")
antispoof_label.grid(row=1, column=2, sticky="sw", padx=10)

liveness_label = ctk.CTkLabel(
    main_frame, text="Liveness: False", font=("Apple Garamond", font_size_total), text_color="#FF0000")
liveness_label.grid(row=2, column=2, sticky="sw", padx=10)

# Format the gesture challenge list
formatted_gesture_challenge = ', '.join(
    [gesture.replace('_', ' ') for gesture in gr.gesture_challenge_list])

# Create the label with the formatted text
challenge_label = ctk.CTkLabel(
    main_frame, text="Challenge: " + formatted_gesture_challenge, font=("Apple Garamond", 30))
challenge_label.grid(row=3, column=1, sticky="sw", padx=10)

# STATUS
status_label = ctk.CTkLabel(main_frame, text="", font=("Arial", font_size))
# Adjust grid positioning as needed
status_label.grid(row=5, column=0, padx=10, columnspan=2)


# BUTTON FRAME
button_frame = ctk.CTkFrame(main_frame, fg_color='#FFF8E9')
button_frame.grid(row=6, column=0, columnspan=3, pady=20)
main_interface()


################### LOOP ###################
update_frame()
app.mainloop()
cap.release()
