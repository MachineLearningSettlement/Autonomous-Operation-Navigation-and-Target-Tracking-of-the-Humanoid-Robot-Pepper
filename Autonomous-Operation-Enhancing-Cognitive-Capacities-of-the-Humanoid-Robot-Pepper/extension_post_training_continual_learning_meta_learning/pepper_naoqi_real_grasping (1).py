import time
import threading
from naoqi import ALProxy
import sys

# Adresse IP et port de Pepper
ip = Pepper_API
port = Pepper_PORT
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
            f.write("P5\n{} {}\n255\n".format(image_width_depth, image_height_depth))
            f.write(image_array_depth)

        print("Image de profondeur enregistrée sous {}".format(depth_filename))

    except Exception as e:
        print("Erreur lors de la capture d'image : {}".format(e))


# ============================================================
# REAL PEPPER GRASPING VIA NAOQI
# ============================================================

JOINT_ANGLES_FILE = "pepper_joint_angles.txt"
GRASPING_STATUS_FILE = "grasping_status.txt"

# Pepper's arm has 5 controllable arm joints.
# The file produced by the orchestration script may contain more
# values; only a complete 5-joint arm configuration is accepted.
RIGHT_ARM_JOINTS = [
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw"
]

LEFT_ARM_JOINTS = [
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw"
]

GRASP_HAND_CLOSE_ANGLE = 1.0
MOTION_SPEED = 0.15
HAND_SPEED = 0.2


def read_ik_joint_angles(filename=JOINT_ANGLES_FILE):
    """Read IK joint values written by the orchestration script."""
    try:
        angles = []

        with open(filename, "r") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                if "=" in line:
                    _, value = line.split("=", 1)
                else:
                    value = line

                angles.append(float(value.strip()))

        if len(angles) < 5:
            print("ERROR: IK file does not contain a complete 5-joint Pepper arm configuration.")
            return None

        # Pepper arm control requires the five arm joints.
        return angles[:5]

    except Exception as e:
        print("Error reading IK joint angles:", e)
        return None


def verify_joint_configuration(motion_proxy, joint_names, target_angles, tolerance=0.08):
    """Verify that Pepper reached the commanded joint configuration."""
    try:
        current_angles = motion_proxy.getAngles(joint_names, True)

        if len(current_angles) != len(target_angles):
            return False

        for current, target in zip(current_angles, target_angles):
            if abs(current - target) > tolerance:
                print(
                    "Joint verification failed: current={} target={}".format(
                        current, target
                    )
                )
                return False

        return True

    except Exception as e:
        print("Error verifying joint configuration:", e)
        return False


def execute_real_grasping(ip, port):
    """
    Execute the grasp physically on Pepper through NAOqi.

    Sequence:
      1. Read the validated IK joint angles.
      2. Move the selected arm to the IK configuration.
      3. Verify that the arm reached that configuration.
      4. Close the corresponding Pepper hand.
      5. Verify the hand closed.
      6. Write exactly 'end grasping' only after all steps succeed.
    """
    status_written = False

    try:
        motion_proxy = ALProxy("ALMotion", ip, port)

        # Read IK output generated by the previous orchestration stage.
        target_angles = read_ik_joint_angles()

        if target_angles is None:
            print("GRASPING ABORTED: invalid IK configuration.")
            return False

        # The external execution script uses the right arm by default.
        # This is intentionally explicit so the IK joint ordering is not hidden.
        arm_joints = RIGHT_ARM_JOINTS
        hand_joint = "RHand"

        print("REAL GRASPING -> connecting to Pepper ALMotion...")
        print("REAL GRASPING -> joints:", arm_joints)
        print("REAL GRASPING -> IK angles:", target_angles)

        # Make the arm safe to command.
        motion_proxy.setStiffnesses(arm_joints, 1.0)

        # Move Pepper's arm to the IK solution.
        motion_proxy.angleInterpolation(
            arm_joints,
            target_angles,
            [1.0] * len(arm_joints),
            True
        )

        print("REAL GRASPING -> arm reached IK configuration.")

        # Verify the actual joint positions before closing the hand.
        if not verify_joint_configuration(
            motion_proxy,
            arm_joints,
            target_angles
        ):
            print("GRASPING ABORTED: Pepper did not reach the IK configuration.")
            return False

        # Close Pepper's hand physically through NAOqi.
        print("REAL GRASPING -> closing Pepper hand...")
        motion_proxy.setAngles(
            hand_joint,
            GRASP_HAND_CLOSE_ANGLE,
            HAND_SPEED
        )

        time.sleep(1.5)

        # Verify the hand reached its commanded closed position.
        hand_angle = motion_proxy.getAngles(hand_joint, True)[0]

        if abs(hand_angle - GRASP_HAND_CLOSE_ANGLE) > 0.12:
            print(
                "GRASPING ABORTED: Pepper hand did not reach closed position "
                "(current={})".format(hand_angle)
            )
            return False

        print("REAL GRASPING -> Pepper completed the commanded grasp.")

        # Only now signal the orchestration process.
        with open(GRASPING_STATUS_FILE, "w") as file:
            file.write("end grasping")

        status_written = True
        print("GRASPING STATUS -> end grasping")

        return True

    except Exception as e:
        print("REAL GRASPING ERROR:", e)
        return False

    finally:
        if status_written:
            print("Grasping finished successfully.")
        else:
            print("Grasping not completed; status file was not updated.")


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
