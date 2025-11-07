import os
import google.generativeai as genai
from PIL import Image

API_KEY = ""
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

print("Please note, if you would like to end the convesation, enter exit to quit")

img_path = "images/Wellspicture.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

image = Image.open(img_path)

try:
    first_msg = [
        "How many number wells are there total, and how many are filled in?",
        image
    ]
    wellsnum = chat.send_message(first_msg)
    print(wellsnum.text.strip() if wellsnum.text else "(No text response)")
except Exception as e:
    print(f"Error on first message: {e}")


while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = chat.send_message(user_input)
    print("Gemeni:", response.text)




# while True:
#     try:
#         user_input = input("You: ").strip()
#         if user_input.lower() == "exit":
#             break
#         response = chat.send_message(user_input)
#         print("Gemini:", response.text.strip() if response.text else "(No text response)")
#     except KeyboardInterrupt:
#         print("\nExiting.")
#         break
#     except Exception as e:
#         print(f"Error: {e}")
