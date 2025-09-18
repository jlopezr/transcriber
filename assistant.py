import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY  = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
MODEL    = os.getenv("LMSTUDIO_MODEL", "gemma-3-27b-it")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Mensajes “persistentes” del chat
messages = [
    {"role": "system", "content": "Eres un asistente útil, conciso y en español."}
]

def chat_once(prompt: str, stream: bool = True):
    messages.append({"role": "user", "content": prompt})
    if stream:
        with client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            top_p=0.95,
            max_tokens=800,
            stream=True,
        ) as s:
            sys.stdout.write("Asistente: ")
            full = []
            for chunk in s:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()
                    full.append(delta.content)
            sys.stdout.write("\n")
            assistant_msg = "".join(full)
    else:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            top_p=0.95,
            max_tokens=800,
        )
        assistant_msg = resp.choices[0].message.content
        print("\nAsistente:", assistant_msg)

    messages.append({"role": "assistant", "content": assistant_msg})

def main():
    print(f"Conectando a {BASE_URL} con modelo: {MODEL}")
    print("Escribe 'salir' para terminar.\n")
    while True:
        try:
            user = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChao!")
            break
        if user.lower() in {"salir", "exit", "quit"}:
            print("Chao!")
            break
        if not user:
            continue
        chat_once(user, stream=True)

if __name__ == "__main__":
    main()
