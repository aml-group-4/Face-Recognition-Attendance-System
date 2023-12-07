import customtkinter as ctk
from customtkinter import FontManager
import subprocess

app = ctk.CTk()
app.title("Face Recognition Attendance System")
app.geometry("1440x810")
ctk.set_appearance_mode("light")

# Load the custom font
FontManager.load_font("AppleGaramond.ttf")

def create_button(parent, text, command, fg_color, hover_color, text_color):
    return ctk.CTkButton(parent, text=text, command=command, fg_color=fg_color, 
                         text_color=text_color, hover_color=hover_color, font=("Arial", font_size), 
                         width=button_width, height=button_height, 
                         corner_radius=corner_radius, border_color=border_color, 
                         border_width=border_width)

def unsupervised_model():
    pass

def supervised_model ():
    subprocess.Popen(["python", "supervised.py"])
    app.quit()
    

def exit_app():
    app.quit()

main_frame = ctk.CTkFrame(app, fg_color='#FFF8E9')
main_frame.pack(expand=True, fill='both')

button_width, button_height, font_size = 460, 70, 48
corner_radius, border_color, border_width = 12, "black", 2

# Title Label inside main_frame
title_label = ctk.CTkLabel(main_frame, text="Face Recognition Attendance System", font=("Apple Garamond", 84))
title_label.pack(pady=(50, 100))

# Buttons inside main_frame
buttonClassificationSupervised = create_button(main_frame, "Supervised model 1", supervised_model, "#FFC300", "#E9B200", "black")
buttonClassificationSupervised.pack(pady=10)

buttonTripletSupervised = create_button(main_frame, "Supervised model 2", supervised_model, "#FFC300", "#E9B200", "black")
buttonTripletSupervised.pack(pady=10)

buttonUnsupervised = create_button(main_frame, "Unsupervised model", unsupervised_model, "#204BA4", "#022367", "white")
buttonUnsupervised.pack(pady=10)

buttonExit = create_button(main_frame, "Exit", exit_app, "#F22C17", "#BF1200", "white")
buttonExit.pack(pady=10)

app.mainloop()