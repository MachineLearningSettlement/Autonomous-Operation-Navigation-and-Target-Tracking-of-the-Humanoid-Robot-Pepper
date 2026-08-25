from PIL import Image
import os
#pour bien comprendre le backend voir DLIB_Reconnaissance faciale incrémentale
import numpy as np
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

if __name__ == "__main__":
    
    # Chemin vers le fichier PPM
    ppm_file = "pepper_image.ppm"
    # Dossier de sortie et nom du fichier
    output_folder = "PepperImages"
    output_filename = "pepper_image.png"
    # Sauvegarder et visualiser l'image
    save_and_visualize_ppm(ppm_file, output_folder, output_filename)
       
      
        
    
      
