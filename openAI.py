import openai
import os
openai.api_key = "YOUR_API_KEY_HERE"
messages = [
    {"role": "system", "content": "You are solving math problems. do not show any steps, just return a one line answer "
                                  "to me. if it is not a math problem return an error 'Error: not a problem'"}
]
def entry(message):
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    reply = chat.choices[0].message.content
    return f"ChatGPT:{reply}"
    #messages.append({"role": "user", "content": message})
