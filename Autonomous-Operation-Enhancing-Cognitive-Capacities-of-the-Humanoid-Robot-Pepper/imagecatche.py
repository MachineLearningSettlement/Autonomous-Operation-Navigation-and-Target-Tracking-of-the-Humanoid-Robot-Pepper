import sys
from naoqi import ALProxy
import time  # Pour ajouter un délai entre les captures d'images

def set_head_position(motion_proxy, head_yaw, head_pitch):
    """Set the head position of Pepper."""
    motion_proxy.setAngles("HeadYaw", head_yaw, 0.2)
    motion_proxy.setAngles("HeadPitch", head_pitch, 0.2)
    time.sleep(2)  # Wait for the motion to complete

def capture_image(ip, port):
    try:
        # Créer un proxy pour le service ALMotion
        motion_proxy = ALProxy("ALMotion", ip, port)

        # Set the head position before capturing the image
        set_head_position(motion_proxy, -0.0813, 0.4418)   #angles = [-0.0337, 0.4387]  # [HeadYaw, HeadPitch]


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
    # Adresse IP et port du robot Pepper
    pepper_ip = "10.12.20.165"  # Remplacez par l'adresse IP de votre robot Pepper
    pepper_port = 9559

    try:
        while True:
            capture_image(pepper_ip, pepper_port)
            time.sleep(2)  # Pause de 2 secondes entre chaque capture
    except KeyboardInterrupt:
        print("\nCapture interrompue par l'utilisateur.")
