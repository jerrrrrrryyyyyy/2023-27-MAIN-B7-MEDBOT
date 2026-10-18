import requests

SERVER_URL = "http://127.0.0.1:8000/chat"


def main():
    print("Health Chatbot (type 'quit' to exit)")
    print("-" * 40)

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = requests.post(SERVER_URL, json={"message": user_input})
            response.raise_for_status()
            reply = response.json()["reply"]
            print(f"Bot: {reply}")
        except requests.exceptions.ConnectionError:
            print("Error: Could not reach the server. Is main.py running?")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()