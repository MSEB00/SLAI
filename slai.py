import json
import logging
import subprocess
import pyttsx3
import spacy
from fuzzywuzzy import process
from datetime import datetime
import sys
import psutil

import memory
from neural_net import predict_intent

# Setup logging
logging.basicConfig(
    filename='slai.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

engine = pyttsx3.init()
nlp = spacy.load("en_core_web_sm")

# Load apps
with open('apps.json', 'r') as f:
    apps = json.load(f)

def speak(text):
    print(f"SLAI: {text}")
    logging.info(f"SLAI: {text}")
    engine.say(text)
    engine.runAndWait()

def open_app(app_name):
    logging.info(f"Attempting to open app: {app_name}")
    match, score = process.extractOne(app_name.lower(), apps.keys(), score_cutoff=85)

    if score:
        command = apps[match]
        try:
            # Platform-specific command execution
            if sys.platform == "win32":
                if command.lower().endswith(".exe"):
                    subprocess.Popen(command)
                else:
                    subprocess.Popen(['start', '', command], shell=True)
                speak(f"Opening {match}")
            elif sys.platform == "darwin":
                subprocess.Popen(['open', '-a', command])
                speak(f"Opening {match}")
            else:
                subprocess.Popen([command])
                speak(f"Opening {match}")
            logging.info(f"Successfully opened app: {match} with fuzzy score {score}")
            return True
        except FileNotFoundError:
            logging.error(f"Command not found for {match} ({command}). Ensure it's in PATH or provide full path.")
            speak(f"Sorry, I couldn't find the command to open {match}.")
            return False
        except Exception as e:
            logging.error(f"Failed to open {match} using command '{command}': {e}")
            speak(f"Failed to open {match}: {str(e)}")
            return False
    else:
        logging.info(f"No sufficiently close match found for '{app_name}'. Best match: {match} with score {score}")
        return False
def get_app_name_from_input(user_input):
    doc = nlp(user_input)
    
    potential_app_name = None

    # Strategy 1: Look for noun chunks that are direct objects of "open"
    for token in doc:
        if token.lemma_ == "open" and token.pos_ == "VERB":
            # Check for a direct object (dobj) or a child that looks like an app name
            for child in token.children:
                if child.dep_ == "dobj" or (child.pos_ in ["NOUN", "PROPN"] and not child.is_stop):
                    potential_app_name = child.text
                    break # Take the first direct object/noun
            if potential_app_name:
                break
        
    if not potential_app_name:
        # Strategy 2: Look for noun chunks in general, potentially multiple words
        # Filter out very short chunks or common stop words
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if chunk_text not in ["it", "this", "that", "app", "game"] and len(chunk_text) > 2:
                # Prioritize noun chunks that are capitalized (proper nouns)
                if chunk.text.istitle() or chunk.text.isupper():
                    potential_app_name = chunk.text
                    break # Take the first capitalized noun chunk
                elif " " in chunk.text: # Take the first multi-word noun chunk as a strong candidate
                    potential_app_name = chunk.text
                    break
                else: # Fallback to single word noun chunk
                    potential_app_name = chunk.text
                    break
    
    # Strategy 3: Check for specific entities that might be apps (like PRODUCTS, ORGs)
    if not potential_app_name:
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']: 
                potential_app_name = ent.text
                break

    logging.info(f"Extracted potential app name: {potential_app_name}")
    return potential_app_name

def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {}
    for ent in doc.ents:
        # Filter entities that are unlikely to be memory items for now
        # We're keeping PERSON, ORG, GPE, LOC, DATE, TIME
        # You might also want to exclude "PRODUCT" if they are often app names and handled by open_app
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'TIME']:
            key = ent.label_.lower()
            entities.setdefault(key, []).append(ent.text)
    logging.info(f"Extracted entities: {entities}")
    return entities

def handle_memory_statements(entities):
    responses = []
    for etype, values in entities.items():
        for val in values:
            memory.remember_entity(etype, val)
            responses.append(f"I'll remember your {etype} is {val}.")
            logging.info(f"Remembered {etype}: {val}")
    if responses:
        speak(" ".join(responses))
        return True
    return False

def handle_memory_queries(user_input):
    lowered = user_input.lower()
    if any(phrase in lowered for phrase in ["what is my name", "who am i", "do you know my name"]):
        names = memory.recall_entity('person')
        if names:
            speak(f"You told me your name is {names[-1]}.")
        else:
            speak("I don't know your name yet.")
        return True
    if any(phrase in lowered for phrase in ["where do i live", "where am i from", "my location"]):
        locations = memory.recall_entity('gpe') + memory.recall_entity('loc')
        if locations:
            speak(f"You said you live in {locations[-1]}.")
        else:
            speak("I don't know where you live yet.")
        return True
    if any(phrase in lowered for phrase in ["when is my birthday", "my birthday"]):
        dates = memory.recall_entity('date')
        if dates:
            speak(f"You told me your birthday is {dates[-1]}.")
        else:
            speak("I don't know your birthday yet.")
        return True
    return False

def main():
    speak("Welcome back, MSEB Gaming.")
    speak("How can I help you today?")
    logging.info("SLAI started.")

    while True:
        try:
            user_input = input("You: ").strip()
            logging.info(f"User: {user_input}")

            if not user_input:
                continue

            lowered = user_input.lower()

            if lowered in ['exit', 'quit', 'bye', 'goodbye', 'farewell', 'see you later']:
                speak("Goodbye!")
                logging.info("SLAI session ended.")
                break

            if lowered in ['thank you', 'thanks', 'appreciate it']:
                speak("You're welcome!")
                continue

            # Hardcoded responses are handled here. Consider making these configurable
            # or integrating with the NLU if they overlap with potential intents.
            if lowered in ['tell me a joke', 'make me laugh', 'say something funny', 'joke please', 'do you know any jokes']:
                speak("Why don't scientists trust atoms? Because they make up everything!")
                continue

            if lowered in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you', "what's up"]:
                speak("Hello! How can I help?")
                continue

            # Check for memory statements
            entities = extract_entities(user_input)
            if entities:
                if handle_memory_statements(entities):
                    continue

            # Check for memory queries
            if handle_memory_queries(user_input):
                continue

            # Predict intent
            intent = predict_intent(user_input)
            logging.info(f"Predicted intent: {intent}")

            if intent == "time":
                now = datetime.now().strftime("%H:%M:%S")
                speak(f"The current time is {now}")

            elif intent == "date":
                today = datetime.now().strftime("%Y-%m-%d")
                speak(f"Today's date is {today}")
            
            elif intent == "mood":
                # Currently, this is a placeholder. Emotion recognition will enhance this.
                speak("I'm here for you. Want to talk about it?")
                continue

            elif intent == "open_app":
                app_name_candidate = get_app_name_from_input(user_input)
                if app_name_candidate:
                    # No need to dynamically add "mecha break" here anymore if it's in apps.json
                    if not open_app(app_name_candidate):
                        speak(f"I couldn't find an app matching '{app_name_candidate}' to open. Please specify or train me.")
                else:
                    speak("I recognized you want to open an app, but I didn't catch which one. Can you be more specific?")
            elif intent == "close_app":
                app_name_candidate = get_app_name_from_input(user_input)
                if app_name_candidate:
                    if not close_app(app_name_candidate):
                        speak(f"I couldn't find an app matching '{app_name_candidate}' to close, or it wasn't running.")
                else:
                    speak("I recognized you want to close an app, but I didn't catch which one. Can you be more specific?")
            else:
                # ... (learning new intents) ...
                pass
            continue

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            speak("Oops, something went wrong.")

if __name__ == "__main__":
    main()

def close_app(app_name):
    # Use fuzzy matching to find the app command
    match, score = process.extractOne(app_name.lower(), apps.keys(), score_cutoff=85)
    if score:
        command = apps[match]
        # Try to find and terminate the process
        try:
            closed = False
            for proc in psutil.process_iter(['name', 'exe', 'cmdline']):
                try:
                    proc_name = proc.info['name'] or ''
                    proc_exe = proc.info['exe'] or ''
                    proc_cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if (command.lower() in proc_name.lower() or
                        command.lower() in proc_exe.lower() or
                        command.lower() in proc_cmdline.lower() or
                        match.lower() in proc_name.lower() or
                        match.lower() in proc_exe.lower() or
                        match.lower() in proc_cmdline.lower()):
                        proc.terminate()
                        closed = True
                        logging.info(f"Closed app: {match}")
                        speak(f"Closed {match}")
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            if not closed:
                logging.info(f"Could not find running process for: {match}")
                return False
            return True
        except Exception as e:
            logging.error(f"Error closing app {match}: {e}")
            speak(f"Failed to close {match}: {str(e)}")
            return False
    else:
        logging.info(f"No sufficiently close match found for '{app_name}' to close.")
        return False