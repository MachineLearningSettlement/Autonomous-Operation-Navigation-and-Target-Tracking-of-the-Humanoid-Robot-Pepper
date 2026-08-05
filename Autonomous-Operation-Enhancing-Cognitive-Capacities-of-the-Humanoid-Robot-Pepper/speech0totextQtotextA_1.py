import speech_recognition as sr
import google.generativeai as genai
from ultralytics import YOLO
import os
from PIL import Image
import time
import os
import face_recognition
#pour bien comprendre le backend voir DLIB_Reconnaissance faciale incrémentale
import numpy as np
import cv2
from datetime import datetime


# Configurez votre clé API pour Google Gemini
genai.configure(api_key="AIzaSyDkDWQpai1hCJ-nusqqZ-nRb0XERY8kunQ")


def speech_to_text(output_file):
    """Convertit la voix en texte et enregistre dans un fichier."""
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        print("Microphone ready. You can speak now!")

        try:
            print("Listening...")
            audio = recognizer.listen(source, phrase_time_limit=60)
            print("Processing your input...")
            text = recognizer.recognize_google(audio, language="en-US")
            print(f"Recognized Text: {text}")

            with open(output_file, "w") as file:
                file.write(text)
            print(f"Text successfully saved to {output_file}")
            return text

        except sr.WaitTimeoutError:
            print("No speech detected. Please try again.")
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    
def ask_gemini(question, output_file):
    """Envoie une question au modèle Gemini et enregistre la réponse."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
                f"Imagine you are Pepper, a humanoid robot. Respond directly to the user's input: {question} in ENGLISH, not English, without exceeding 100 words. Avoid repeating 'Hello' in every answer. If the {question} relates to something like 'stop', 'I don't understand what you're saying', or anything suggesting the user is interrupting the robot for saying something ridiculous, return exactly this response: 'quite'."
                    )

        bot_response = response.text  # Ajustez si nécessaire
        print(f"Pepper's response: {bot_response}")

        # Enregistrer la réponse dans un fichier texte
        with open(output_file, "w") as file:
            bot_response = bot_response.lower() #ensure exactly that quite is quite and not Quite
            file.write(bot_response)
        print(f"Response successfully saved to {output_file}")

    except Exception as e:
        print(f"Error generating response: {e} for ask_gemini ")

def object_detection(file_path, output_folder, output_filename, image_path, output_file):
    # Ouvrir l'image PPM avec Pillow
    image = Image.open(file_path)

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Chemin complet du fichier de sortie
    output_path = os.path.join(output_folder, output_filename)

    # Sauvegarder l'image au format PNG
    image.save(output_path, "PNG")

    # Visualiser l'image
    image.show()
    
    
    model_yolo = YOLO("yolov8s.pt")

    # Effectuer la détection des objets
    results = model_yolo.predict(source=image_path, conf=0.5)

    # Vérifier si des objets ont été détectés
    if results[0].boxes:
        detected_objects_with_conf = [
            (model_yolo.names[int(cls)], float(conf))  
            for cls, conf in zip(results[0].boxes.cls, results[0].boxes.conf)
        ]
    else:
        detected_objects_with_conf = []

    # Écrire les objets détectés dans un fichier
    with open("objects_detection.txt", "w") as file:
        for obj, conf in detected_objects_with_conf:
            file.write(obj + ", ")
            print(f"Objet détecté : {obj}, Confiance : {conf:.2f}")

    # Lire le contenu du fichier et générer la réponse
    try:
        with open("objects_detection.txt", "r") as file:
            objects = file.read()
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                    f"Imagine you are Pepper, a humanoid robot. Suppose you have observed these objects with your camera: {objects}. I ask you to provide a concise description of what you see, based on the objects detected by your camera. The description must be in ENGLISH. Also, avoid repeating 'Hello' in each response. If you find that the set of objects is empty, return exactly this sentence: 'I cannot determine the nature of the objects I am currently seeing'."
                        )

            bot_response = response.text
            print(f"Pepper's response: {bot_response}")
            
        if bot_response:
            with open(output_file, "w") as file:
                file.write(bot_response.lower())
                print(f"Response successfully saved to {output_file}")
        else:
            print("No response to save.")
            
    except Exception as e:
        print(f"Error reading the file: {e} for object_detection")
        

def save_and_visualize_ppm(file_path, output_folder, output_filename):
    # Ouvrir l'image PPM avec Pillow
    image = Image.open(file_path)

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Chemin complet du fichier de sortie
    output_path = os.path.join(output_folder, output_filename)

    # Sauvegarder l'image au format PNG
    image.save(output_path, "PNG")

    # Visualiser l'image
    image.show()
    
def incremental_learning_v1(ppm_file, output_folder, output_file) :
    
    # Chemin vers le fichier PPM
    #ppm_file = "pepper_image.ppm"

    # Dossier de sortie et nom du fichier
    #output_folder = "face_recog_base"
    model = genai.GenerativeModel("gemini-2.5-flash")

    response_person = model.generate_content(
            f"Imagine you are Pepper, a humanoid robot. Take this sentence: {user_question}, and return ONLY the name of the person being introduced to you — NOTHING MORE, NOTHING LESS."
            )
    var = response_person.text.lower().strip()
    print('var : ', var )
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_filename = f"{var}_{time}.png"
    print('output_filename : ', output_filename )

    # Sauvegarder et visualiser l'image
    save_and_visualize_ppm(ppm_file, output_folder, output_filename)
        
    # Charger ton visage enregistré
    output_path = os.path.join('face_recog_base', output_filename)
    image_base = face_recognition.load_image_file(output_path)  # Ton image de référence  
    encoding = face_recognition.face_encodings(image_base)  # Extraire l'embedding

    if len(encoding) > 0:
        encoding_base = encoding[0]
        # Charger l'image PNG à tester
        folder_path = r"C:\Users\hp\Desktop\Master IA2S\Mini Projet\pythoncode\face_recog_base"  # Modifier avec le nom du fichier à tester

        radar = 0
        for filename in os.listdir(folder_path):
                
            exception = output_filename

        
            if filename!= exception :
                
                image_path = os.path.join(folder_path, filename)  
                test_image = face_recognition.load_image_file(image_path)

                # Extraire les visages détectés
                face_encodings = face_recognition.face_encodings(test_image)

                if len(face_encodings) > 0:
                    test_encoding = face_encodings[0]  # Prendre le premier visage détecté

                    # Comparer avec l'image enregistrée
                    match = face_recognition.compare_faces([encoding_base], test_encoding)
                    distance = face_recognition.face_distance([encoding_base], test_encoding)[0]
    
                    if match[0]:

                        radar = radar + 1

                        time = datetime.now().second


                        if time <= 15 :
                            response = f"Hello {var}! Nice to see you again! How are you doing?"
                        elif 15 < time <= 30 :
                            response = f"Hello {var}! It's a pleasure to see you again! How have you been?"
                        elif 30 < time <= 45 :
                            response = f"Hello {var}! Happy to see you again! I hope you're doing well?"
                        else :
                            response = f"Hello {var}! What a pleasure to see you again! How is your day going?"
                            
                        bot_response = response
                        
                        print(f"Pepper's response: {bot_response}")
                            
                        #print(f"Bonjour {var}, oui je te connais, ravis de te revoir ! (Distance: {distance:.2f})")
                        if bot_response:
                            with open(output_file, "w") as file:
                                file.write(bot_response.lower())
                                print(f"Response successfully saved to {output_file}")
                        else:
                            print("No response to save. incremental_learning")
                        
                        break
                        
                else:
                    print("Je ne vois pas bien votre visage1.")
                        
        if radar == 0 :
            
            time = datetime.now().second

            if time <= 15:
                response = f"Hello {var}, I don't know you, but it's a pleasure to meet you."
            elif 15 < time <= 30:
                response = f"Hello {var}, unfortunately I don't know you, but I'm happy to get to know you."
            elif 30 < time <= 45:
                response = f"Hello {var}, I don't know you, nice to meet you, I'm glad to get to know you."
            else:
                response = f"Hello {var}, actually I don't know you, but it's a pleasure to get to know you."

            bot_response = response
                        
            print(f"Pepper's response: {bot_response}")

            
            #print(f"Bonjour {var}, enchanté, je suis ravis de te connaitre (Distance: {distance:.2f})")

            if bot_response:
                with open(output_file, "w") as file:
                    file.write(bot_response.lower())
                    print(f"Response successfully saved to {output_file}")
            else:
                print("No response to save. incremental_learning")
                
    else:   
        print("I don't see your face0")

def incremental_learning_v2(ppm_file, output_folder, output_file) :
    
    # Chemin vers le fichier PPM
    #ppm_file = "pepper_image.ppm"

    # Dossier de sortie et nom du fichier
    #output_folder = "face_recog_base"

    output_filename = f"connu_inconnu.png"
    print('output_filename : ', output_filename )

    # Sauvegarder et visualiser l'image
    save_and_visualize_ppm(ppm_file, output_folder, output_filename)
        
    # Charger ton visage enregistré
    output_path = os.path.join('face_recog_base', output_filename)
    image_base = face_recognition.load_image_file(output_path)  # Ton image de référence  
    encoding = face_recognition.face_encodings(image_base)  # Extraire l'embedding

    if len(encoding) > 0:
        encoding_base = encoding[0]
        # Charger l'image PNG à tester
        folder_path = r"C:\Users\hp\Desktop\Master IA2S\Mini Projet\pythoncode\face_recog_base"  # Modifier avec le nom du fichier à tester

        radar = 0
        for filename in os.listdir(folder_path):
                
            exception = output_filename

        
            if filename!= exception :
                
                image_path = os.path.join(folder_path, filename)  
                test_image = face_recognition.load_image_file(image_path)

                # Extraire les visages détectés
                face_encodings = face_recognition.face_encodings(test_image)

                if len(face_encodings) > 0:
                    test_encoding = face_encodings[0]  # Prendre le premier visage détecté

                    # Comparer avec l'image enregistrée
                    match = face_recognition.compare_faces([encoding_base], test_encoding)
                    distance = face_recognition.face_distance([encoding_base], test_encoding)[0]
    
                    if match[0]:

                        radar = radar + 1

                        var = filename.split("_")[0]

                        model = genai.GenerativeModel("gemini-2.5-flash")

                        time = datetime.now().second


                        if time <= 15:
                            response = f"Yes, I know you well, your name is {var}."
                        elif 15 < time <= 30:
                            response = f"Yes, I recognize you, your name is {var}."
                        elif 30 < time <= 45:
                            response = f"Yes, I remember you, your name is {var}."
                        else:
                            response = f"Of course, you are {var}, I recognize you!"

                        bot_response = response

                        
                        print(f"Pepper's response: {bot_response}")
                            
                        #print(f"Bonjour {var}, oui je te connais, ravis de te revoir ! (Distance: {distance:.2f})")
                        if bot_response:
                            with open(output_file, "w") as file:
                                file.write(bot_response.lower())
                                print(f"Response successfully saved to {output_file}")
                        else:
                            print("No response to save. incremental_learning")
                        
                        break
                        
                else:
                    print("Je ne vois pas bien votre visage1.")
                    
                        
        if radar == 0 :
            model = genai.GenerativeModel("gemini-2.5-flash")
            time = datetime.now().second

            if time <= 15:
                response = f"No, I don't know you. Could you please introduce yourself?"
            elif 15 < time <= 30:
                response = f"No, we don't know each other yet. Could you introduce yourself?"
            elif 30 < time <= 45:
                response = f"Sorry, I don't know you. Could you tell me who you are?"
            else:
                response = f"I don't believe I know you. Could you please introduce yourself?"
               
    

            

            bot_response = response
                        
            print(f"Pepper's response: {bot_response}")

            
            #print(f"Bonjour {var}, enchanté, je suis ravis de te connaitre (Distance: {distance:.2f})")

            if bot_response:
                with open(output_file, "w") as file:
                    file.write(bot_response.lower())
                    print(f"Response successfully saved to {output_file}")
            else:
                print("No response to save. incremental_learning")
                
    else:   
        print("Je ne vois pas bien votre visage0")

if __name__ == "__main__":
    
    speech_output_filename = "speech_to_text_output.txt"
    response_output_filename = "pepper_response.txt"
    ppm_file = "pepper_image.ppm"
    output_folder = "PepperImages"
    output_folder1 = "face_recog_base"
    output_filename = "pepper_image.png"
    image_path = r"C:\Users\hp\Desktop\Master IA2S\Mini Projet\pythoncode\PepperImages\pepper_image.png"
    
    start_time = time.time()

    while True:
        
        print("Step 1: Converting speech to text...")
        user_question = speech_to_text(speech_output_filename)

        if user_question:
            
            
            print("Step 2: Sending the question to Gemini...")
            
            model = genai.GenerativeModel("gemini-2.5-flash")

            #BLOCK1
            BLOCK1 = model.generate_content(
                f"Imagine you are Pepper, a humanoid robot. If the client's question '{user_question}' concerns the use of cameras or visual analysis of the environment, respond exactly with 'Yes', otherwise respond exactly with 'No' — NOTHING MORE, NOTHING LESS. VERY IMPORTANT NOTE: If '{user_question}' corresponds to an introduction like 'Hello, I am...' or 'My name is...', or a similar phrasing, respond exactly with 'No'."
                    )
            BLOCK1 = 'No'
            
            
            #BLOCK2
            BLOCK2 = model.generate_content(
                      f"If '{user_question}' is similar to 'Do you know me?' or 'Have we talked before?' or 'Have we met before?' OR SOMETHING SIMILAR, respond exactly with 'Yes', otherwise respond exactly with 'No' — NOTHING MORE, NOTHING LESS."
                
                  )
            BLOCK2 = 'Yes'
            
            
            BLOCK3
            BLOCK3 = model.generate_content(
                f"Imagine you are Pepper, a humanoid robot. If '{user_question}' matches exactly a way a person introduces themselves like 'Let me introduce myself, I am...' or 'I am...' or 'Let me introduce myself, my name is...' or 'My name is...', respond exactly with 'Yes', otherwise respond exactly with 'No' — NOTHING MORE, NOTHING LESS."
                   )
            BLOCK3 = 'No'
            
           

            print('Utilisation caméra : ' , BLOCK1)
            with open('cam.txt', "w") as file:
                file.write(BLOCK1)
                
            print('Personne se présente : ' , BLOCK3)
            
            with open('pres.txt', "w") as file:
                file.write(BLOCK3)
                
            print('Personne demande est-ce que tu le connais ou pas : ' , BLOCK2)
            with open('know.txt', "w") as file:
                file.write(BLOCK2)

            #BLOCK 1 :        
            if BLOCK1 == 'Yes':
          
                #current_time = time.time() - start_time
                
                #if current_time > 5 :
                object_detection(ppm_file, output_folder, output_filename, image_path, response_output_filename)
                
            #BLOCK 2 :
            elif BLOCK2 == 'Yes':
                 
                #current_time = time.time() - start_time

                #if current_time > 5 :
                incremental_learning_v2(ppm_file, output_folder1, response_output_filename)
                

            #BLOCK 3 :
            elif BLOCK3 == 'Yes':
    
                #current_time = time.time() - start_time

                #if current_time > 5 :
                incremental_learning_v1(ppm_file, output_folder1, response_output_filename)
                
            
            #BLOCK 4 :
            else :
                
                ask_gemini(user_question, response_output_filename)
                
        else:
            print("No question detected. Continuing...")

        time.sleep(0.3)  # Petite pause pour éviter un trop grand nombre de requêtes
