import time
import threading
from naoqi import ALProxy
import sys

# Adresse IP et port de Pepper
ip = "10.12.20.165"
port = 9559
#stop_event = threading.Event()  # Événement global pour arrêter le discours
old_response = ''
var = 0
# Connexion au proxy de Pepper pour la synthèse vocale
try:
    tts = ALProxy("ALTextToSpeech", ip, port)
    print("Connected to Pepper successfully!")
except Exception as e:
    print("Failed to connect to Pepper:", e)
    exit(1)

def read_from_file(filename):
    """Lit le contenu du fichier texte."""
    try:
        with open(filename, "r") as file:
            return file.read().strip()
    except Exception as e:
        print("Error reading the file:", e)
        return None

def synthesize_speech(text):
    """Fait parler Pepper avec le texte donné et affiche 'je parle'."""
    try:
        print("Pepper says:", text)
        tts.post.say(text)  # Lancement du texte en mode asynchrone
        # Estimation du temps de parole en fonction du nombre de mots
        estimated_speech_time = max(1, len(text.split()) / 2.5)  # ~2.5 mots/sec
        start_time = time.time()

        while time.time() - start_time < estimated_speech_time :  # Attente du temps estimé 
            new_response = read_from_file('pepper_response.txt')
            
            if new_response :
                if new_response == 'quite':
                    tts.stopAll()
                    print('Pepper is quite 0')
                    break
                elif new_response != text and new_response != 'quite'  :
                    tts.stopAll()
                    print('Pepper is quite 1')
                    break
                                                                             
    except Exception as e:
        print("Error while making Pepper speak:", e)


def set_head_position_cam(motion_proxy, head_yaw, head_pitch):
    """Set the head position of Pepper."""
    motion_proxy.setAngles("HeadYaw", head_yaw, 0.2)
    motion_proxy.setAngles("HeadPitch", head_pitch, 0.2)
    time.sleep(2)  # Wait for the motion to complete

def set_head_position_pres(motion_proxy, head_yaw, head_pitch):
    """Set the head position of Pepper."""
    motion_proxy.setAngles("HeadYaw", head_yaw, 0.2)
    motion_proxy.setAngles("HeadPitch", head_pitch, 0.2)
    time.sleep(2)  # Wait for the motion to complete

def capture_image(ip, port):
    
    #with open('cam.txt', 'r') as file :
        #txt_cam = file.read()
    #with open('pres.txt', 'r') as file :
        #txt_pres = file.read()
        
    try:
        # Créer un proxy pour le service ALMotion
        motion_proxy = ALProxy("ALMotion", ip, port)

        #if txt_cam == 'Oui' :
            # Set the head position before capturing the image
            #set_head_position_cam(motion_proxy, -0.0813, 0.4418)   #angles = [-0.0337, 0.4387]  # [HeadYaw, HeadPitch]

        #if txt_pres == 'Oui' :
            # Set the head position before capturing the image
            #set_head_position_pres(motion_proxy, , )   

        # Créer un proxy pour le service ALVideoDevice
        video_proxy = ALProxy("ALVideoDevice", ip, port)

        # Configuration de la caméra
        resolution = 2  # kVGA (640x480) pour l'image RGB
        color_space = 11  # RGB pour l'image couleur
        fps = 10

        # Configuration de la caméra de profondeur
        depth_resolution = 2  # Résolution pour l'image de profondeur
        depth_color_space = 16  # Espace colorimétrique pour la profondeur (depuis la caméra ToF ou le capteur de profondeur)

        # Nom de la caméra (peut être "CameraTop" ou "CameraBottom")
        camera_name_rgb = "CameraTop"  # Caméra RGB
        camera_name_depth = "DepthCamera"  # Nom pour la caméra de profondeur

        # S'inscrire à la caméra RGB
        video_client_rgb = video_proxy.subscribeCamera(camera_name_rgb, 0, resolution, color_space, fps)

        # S'inscrire à la caméra de profondeur
        video_client_depth = video_proxy.subscribeCamera(camera_name_depth, 0, depth_resolution, depth_color_space, fps)

        # Capturer une image RGB
        nao_image_rgb = video_proxy.getImageRemote(video_client_rgb)
        # Capturer une image de profondeur
        nao_image_depth = video_proxy.getImageRemote(video_client_depth)

        # Désinscrire des caméras
        video_proxy.unsubscribe(video_client_rgb)
        video_proxy.unsubscribe(video_client_depth)

        # Extraire les données de l'image RGB
        image_width_rgb = nao_image_rgb[0]
        image_height_rgb = nao_image_rgb[1]
        image_array_rgb = nao_image_rgb[6]

        # Extraire les données de l'image de profondeur
        image_width_depth = nao_image_depth[0]
        image_height_depth = nao_image_depth[1]
        image_array_depth = nao_image_depth[6]

        # Enregistrer l'image RGB au format PPM
        rgb_filename = "pepper_image.ppm"
        with open(rgb_filename, "wb") as f:
            f.write("P6\n{} {}\n255\n".format(image_width_rgb, image_height_rgb))
            f.write(image_array_rgb)

        print("Image RGB enregistrée sous {}".format(rgb_filename))

        # Enregistrer l'image de profondeur
        depth_filename = "pepper_depth_image.ppm"
        with open(depth_filename, "wb") as f:
            f.write("P5\n{} {}\n255\n".format(image_width_depth, image_height_depth))  # P5 pour image grayscale
            f.write(image_array_depth)

        print("Image de profondeur enregistrée sous {}".format(depth_filename))

    except Exception as e:
        print("Erreur lors de la capture d'image : {}".format(e))


if __name__ == "__main__":

    
    while True:
        capture_image(ip, port)
        
        response = read_from_file("pepper_response.txt")
        
        if response :
            if response != old_response:
                print("New response detected:", response)
                synthesize_speech(response)
                old_response = response
        elif not response:
            print("No response to read from the file.")

        time.sleep(1)  # Attente pour limiter les lectures inutiles
