import kagglehub
import shutil
import os

def main():
    # 1. Definir la ruta de destino local
    raw_data_dir = './data/raw'
    os.makedirs(raw_data_dir, exist_ok=True)
    target_file = os.path.join(raw_data_dir, "hotel_bookings.csv")

    print("Conectando con Kaggle para descargar/verificar el dataset...")
    
    # 2. Descargar usando kagglehub (usa caché si ya se descargó antes)
    path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")
    source_file = os.path.join(path, "hotel_bookings.csv")
    
    # 3. Copiar el archivo desde la caché de kaggle a nuestra carpeta del proyecto
    shutil.copy2(source_file, target_file)
    print(f"¡Éxito! El dataset original está guardado en: {target_file}")

if __name__ == "__main__":
    main()