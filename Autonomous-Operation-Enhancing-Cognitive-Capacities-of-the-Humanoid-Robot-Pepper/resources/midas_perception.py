import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai
import os
import cv2

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

  
# Chemin vers le fichier PPM
ppm_file = "pepper_image.ppm"
# Dossier de sortie et nom du fichier
output_folder = "PepperImages"
output_filename = "pepper_image.png"
# Sauvegarder et visualiser l'image
save_and_visualize_ppm(ppm_file, output_folder, output_filename)

# Configurez votre clé API pour Google Gemini
genai.configure(api_key="AIzaSyDk1InqYaF9EFiXxJbeAXDDyWWU0kytVRA")

#request
user_question = "donner moi le petit ballon qui se trouve sur la table"

# Charger MiDaS directement via torch.hub
model_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model_midas.to("cuda" if torch.cuda.is_available() else "cpu")
model_midas.eval()

# Charger YOLOv8
model_yolo = YOLO("yolov8n.pt")  # Utilisez "yolov8s.pt" ou "yolov8m.pt" pour des modèles plus grands

# Charger une image
image_path = r"C:\Users\hp\Desktop\Master IA2S\Mini Projet\pythoncode\PepperImages\pepper_image.png"  # Remplacez par le chemin de votre image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir en RGB

# Convertir l'image NumPy en format PIL
image_pil = Image.fromarray(image)

# Prétraiter l'image pour MiDaS
transform = Compose([ 
    Resize((384, 384)),  # Redimensionner à la taille attendue par MiDaS_small
    ToTensor(),           # Convertir en tenseur PyTorch
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliser
])
input_image = transform(image_pil).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

# Estimer la carte de profondeur avec MiDaS
with torch.no_grad():
    depth = model_midas(input_image)
depth = depth.squeeze().cpu().numpy()
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Détecter les objets avec YOLOv8
results = model_yolo(image)

# Paramètres intrinsèques de la caméra (exemple)
fx, fy = 584, 584    # Distance focale en pixels
cx, cy = 320, 240  # Centre de l'image en pixels

# Matrice de calibration K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])



# Inverse de K
K_inv = np.linalg.inv(K)

# Calibration de la profondeur (exemple : 2 mètres = 500 unités)
distance_reelle = 0.50  # en mètres
Z_reference = 634.88    # profondeur relative correspondante
facteur_echelle = distance_reelle / Z_reference

# Dictionnaire pour stocker les données des objets
objects_data = {}

# Combiner les résultats de YOLOv8 et MiDaS
coords_3d_metres_pepper = np.zeros(3)
L = 0.9
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées de la boîte
        cls_id = int(box.cls)  # ID de la classe
        conf = float(box.conf)  # Confiance de la détection
        label = model_yolo.names[cls_id]  # Nom de la classe

        # Calculer le centre de la boîte
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Redimensionner les coordonnées pour correspondre à la carte de profondeur (384x384)
        scale_x = 384 / image.shape[1]  # Facteur d'échelle en largeur
        scale_y = 384 / image.shape[0]  # Facteur d'échelle en hauteur
        center_x_scaled = int(center_x * scale_x)
        center_y_scaled = int(center_y * scale_y)

        # Estimer la profondeur au centre de la boîte
        depth_value = depth[center_y_scaled, center_x_scaled]  # Profondeur en unités relatives

        # Convertir les coordonnées 2D en coordonnées 3D
        coords_2d = np.array([center_x, center_y, 1])
        coords_3d_relative = depth_value * np.dot(K_inv, coords_2d)

        # Convertir en mètres
        coords_3d_metres_camera = coords_3d_relative * facteur_echelle

        # Transformation du repère caméra vers repère Pepper
        coords_3d_metres_pepper[0] = coords_3d_metres_camera[2]    # X = Z
        coords_3d_metres_pepper[1] = -coords_3d_metres_camera[0]   # Y = -X
        coords_3d_metres_pepper[2] = L - coords_3d_metres_camera[0] # Z = L - Y

        # Afficher les résultats
        print(f"Objet: {label}")
        print("Coordonnées 3D relatives :", coords_3d_relative)
        print("Coordonnées 3D en mètres par rapport au repère caméra :", coords_3d_metres_camera)
        print("Coordonnées 3D en mètres par rapport au repère Pepper :", coords_3d_metres_pepper)
        print("-" * 40)

        # Dessiner la boîte et l'étiquette
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sauvegarder les données dans un fichier text
        with open("Object_3D_Script.txt", "a") as file:
            file.write(f"objet détecté : {label}, Coordonnée X dans espace 3D : { coords_3d_metres_pepper[0]}, Coordonnée Y dans espace 3D : {coords_3d_metres_pepper[1]}, Coordonnée Z dans espace 3D : {coords_3d_metres_pepper[2]}\n")

with open("Object_3D_Script.txt", "r") as file:
    Script =  file.read().strip()
            
    model = genai.GenerativeModel("gemini-1.5-flash")
    response_std = model.generate_content(f"Imagine que tu es Pepper, un robot humanoïde. Prends cette description '{Script}' et extrais uniquement les informations relatives au label ainsi que les coordonnées X, Y, Z de l'objet que le client te demande de saisir ou d'attraper. Voici la demande du client : '{user_question}'. Retourne ces informations sous une forme standarisée comme cela : Objet : label, CordX : X, CordY : Y, CordZ : Z. ATTENTION RETOURNER EXACTEMENT CETTE FORME DE REPONSE ET DE RIEN AJOUTER D'AUTRE COMME REPONSE CA VEUT DIRE JE NE VEUX VOIR COMME OUTPUT DE TA REPONSE QUE CELA")
    
    print(response_std.text)

    with open("Coordonnees_3D_object.txt", "w") as file:
        file.write(response_std.text)
                
  

#supprimer le fichier text pour éviter l'encombrement des infos         
   
os.remove("Object_3D_Script.txt")
            
# Afficher l'image avec les détections
cv2.imshow("Résultats", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
