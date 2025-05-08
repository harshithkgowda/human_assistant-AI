import sqlite3
import pyttsx3
import speech_recognition as sr
import time

# List of known objects and aliases
OBJECT_ALIASES = {
    "cell phone": ["cell phone", "phone", "mobile"],
    "pen": ["pen"],
    "bottle": ["bottle", "water bottle"],
    "toothbrush": ["toothbrush", "brush"],
    "rubik's cube": ["rubik's cube", "cube", "puzzle"]
}

# Flatten all keywords for detection
ALL_KEYWORDS = [alias for aliases in OBJECT_ALIASES.values() for alias in aliases]

# Connect to the database
conn = sqlite3.connect("object_locations.db")
cursor = conn.cursor()

# Initialize TTS and speech recognizer
engine = pyttsx3.init()
recognizer = sr.Recognizer()

def format_color(color_str):
    try:
        color = eval(color_str)
        return f"with RGB color {tuple(color)}"
    except:
        return ""

def format_time_difference(timestamp):
    seconds = int(time.time() - timestamp)
    if seconds < 60:
        return f"{seconds} seconds ago"
    elif seconds < 3600:
        return f"{seconds // 60} minutes ago"
    else:
        return f"{seconds // 3600} hours ago"

def extract_known_object(text):
    text = text.lower()
    for canonical_name, aliases in OBJECT_ALIASES.items():
        for alias in aliases:
            if alias in text:
                return canonical_name
    return None

def find_object_location(object_name):
    cursor.execute("SELECT name, color, location, timestamp FROM objects WHERE name = ?", (object_name,))
    rows = cursor.fetchall()

    if rows:
        name, color, location, timestamp = rows[-1]  # Most recent detection
        color_desc = format_color(color)
        time_ago = format_time_difference(timestamp)
        response = f"Your {name} {color_desc} was last seen {location} about {time_ago}."
    else:
        response = f"Sorry, I couldn't find the last location of your {object_name}."

    print("🗣", response)
    engine.say(response)
    engine.runAndWait()

def wait_for_wake_word():
    print("🎙 Say 'Hey Mark' to activate...")
    while True:
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, phrase_time_limit=4)
                text = recognizer.recognize_google(audio).lower()
                if "hey mark" in text:
                    print("🟢 Activated. Listening for command...")
                    engine.say("Yes, what are you looking for?")
                    engine.runAndWait()
                    return
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                print("⚠️ Speech recognition service unavailable.")
                time.sleep(3)

# Adjust for ambient noise once
with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source)

# Main loop
while True:
    wait_for_wake_word()
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print("📥 You said:", command)

            object_name = extract_known_object(command)
            if object_name:
                find_object_location(object_name)
            else:
                engine.say("I didn't catch what object you're looking for. Please try again.")
                engine.runAndWait()

        except sr.UnknownValueError:
            print("❌ Didn't understand the command.")
        except sr.RequestError:
            print("❌ Speech recognition service is unavailable.")
            engine.say("Speech recognition service is currently unavailable.")
            engine.runAndWait()

conn.close()
