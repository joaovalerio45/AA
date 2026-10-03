#week1

import random


responses = {

    "hello": ["Hello, how are you today?", "Hi, how can I help you?"],
    "bye": ["Bye, see you next time!", "Goodbye!"],
    "thanks": ["You're very welcome!", "Happy to chat!", "Anytime!"],
    "sorry": ["No need to apologize at all.", "It's completely okay."],
    "help": ["How can I assist you?", "What do you need help with?"],

    "happy": ["Why do you feel happy?", "What made you happy?"],
    "sad": ["Why do you feel sad?", "What made you feel sad?"],
    "angry": ["What made you feel so angry?", "Why does that upset you?"],
    "tired": ["Did you not sleep well?", "Why are you feeling so exhausted?"],
    "stressed": ["What is causing you stress right now?", "Take a deep breath. What's on your mind?"],
    "excited": ["That sounds great! What are you most excited about?"],
    "bored": ["What do you usually enjoy doing?", "Why do you feel bored today?"],


    "work": ["How are things going at work?", "Do you enjoy what you do?"],
    "school": ["How are your classes going?", "Is school keeping you busy?"],
    "family": ["Tell me more about your family.", "How are things with your family?"],
    "friend": ["Tell me about your friends.", "Did something happen with a friend?"],

    "yes": ["You seem quite certain.", "Could you elaborate on why yes?"],
    "no": ["Why not?", "What makes you say no?"],
    "why": ["Why do you think that is?", "What is your take on it?"]
}

generic = [
    "Tell me more about that.",
    "I see.",
    "Can you explain further?"
]

def pickResponse(answer):
    answer = answer.lower()
    for keyword, replies in responses.items():
        if keyword in answer:
            return random.choice(replies)
    return random.choice(generic)

print("Hello! How can i help you today? (type exit to leave)")

while True:
    message = input("You:")
    if message.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = pickResponse(message)

    print(f"Chatbot: {response}\n")


