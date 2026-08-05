from ultralytics import YOLO
import cv2

# Charger le modèle YOLO
model = YOLO("yolov8s.pt")

# Charger l'image
image_path = r"C:\Users\hp\Desktop\Master IA2S\Mini Projet\pythoncode\PepperImages\pepper_image.png"
depth_image_path = r"C:\Users\hp\Desktop\Master IA2S\Mini Projet\pythoncode\pepper_depth_image.ppm"

image = cv2.imread(image_path)
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Chargement en tant qu'image de profondeur

# Vérifier si l'image et l'image de profondeur ont été correctement chargées
if image is None:
    raise FileNotFoundError(f"Impossible de charger l'image à l'emplacement : {image_path}")
if depth_image is None:
    raise FileNotFoundError(f"Impossible de charger l'image de profondeur à l'emplacement : {depth_image_path}")

# Obtenir les dimensions de l'image
height, width, channels = image.shape
print(f"Dimensions de l'image - Largeur : {width} pixels, Hauteur : {height} pixels, Canaux : {channels}")
print(f"Dimensions de l'image de profondeur - Largeur : {depth_image.shape[1]} pixels, Hauteur : {depth_image.shape[0]} pixels")

# Effectuer la détection des objets
results = model.predict(source=image_path, conf=0.5)

# Initialisation des listes pour stocker les objets détectés
detected_objects_with_conf = []
detected_objects_with_coords = []
shape = []
shape.append((height, width, channels))

# Vérifier si des objets ont été détectés
if results[0].boxes:
    for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        x_min, y_min, x_max, y_max = map(int, box)  # Convertir en pixels
        center_x = (x_min + x_max) // 2  # Calcul du centre en X
        center_y = (y_min + y_max) // 2  # Calcul du centre en Y
        obj_name = model.names[int(cls)]  # Nom de l'objet détecté
        confidence = float(conf)  # Score de confiance

        # Récupérer la profondeur (Z) à partir de l'image de profondeur
        depth_z = depth_image[center_y, center_x]  # Valeur de la profondeur en fonction des coordonnées (center_x, center_y)

        # Ajouter aux listes
        detected_objects_with_conf.append((obj_name, confidence))
        detected_objects_with_coords.append((obj_name, confidence, center_x, center_y, depth_z))

        # Afficher les résultats
        print(f"Objet détecté : {obj_name}, Confiance : {confidence:.2f}, "
              f"Centre X: {center_x}, Centre Y: {center_y}, Profondeur Z: {depth_z}")

# Sauvegarder les objets détectés (nom + confiance) dans objects_detection.txt
with open("objects_detection.txt", "w") as file:
    for obj, conf in detected_objects_with_conf:
        file.write(f"{obj}, {conf:.2f}\n")

# Sauvegarder les objets détectés avec leurs coordonnées dans objects_coordinates.txt
with open("objects_coordinates.txt", "w") as file:
    for obj, conf, center_x, center_y, depth_z in detected_objects_with_coords:
        # Calculer X, Y, Z en utilisant les formules de calibration si nécessaire
        # Exemple simple : ici, Z est la valeur de profondeur en pixels, il faut une calibration pour obtenir les coordonnées réelles
        X = (center_x - (width / 2)) * depth_z / 2407  # Exemple de calcul de X
        Y = (center_y - (height / 2)) * depth_z / 2407  # Exemple de calcul de Y
        
        # Sauvegarder les coordonnées 3D
        file.write(f"{obj}, {conf:.2f}, {center_x}, {center_y}, {depth_z}, {X}, {Y}, {depth_z}\n")

    for x, y, z in shape:
        file.write(f"height: {x}, width: {y}, channels: {z}\n")
