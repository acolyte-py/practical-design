"""Код для тест-кейса"""
import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from torch import autocast, float16
from diffusers import StableDiffusionPipeline


app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(height=40, width=512, text_color="black", fg_color="white", master=app)
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(height=512, width=512, master=app)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=float16)
pipe.to(device)


def generate_image():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]

    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)


trigger = ctk.CTkButton(height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue", command=generate_image)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
