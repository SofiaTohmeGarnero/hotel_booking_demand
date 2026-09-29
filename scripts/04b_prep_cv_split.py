import os
import pandas as pd

def main():
    print("Cargando datasets limpios...")
    processed_dir = './data/processed'
    train_path = os.path.join(processed_dir, 'train_clean.csv')
    test_path = os.path.join(processed_dir, 'test_clean.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("No se encontraron los archivos en data/processed/. Ejecuta 03_preprocessing.py primero.")
        
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Columnas a eliminar para las matrices de features
    drop_para_modelo = ['is_canceled', 'booking_date', 'arrival_date']
    
    print("Generando matrices X e y para Cross Validation...")
    # Para CV, usamos todo el 85% como conjunto de entrenamiento
    X_train_full = df_train_full.drop(columns=drop_para_modelo)
    y_train_full = df_train_full['is_canceled']
    
    X_test = df_test.drop(columns=drop_para_modelo)
    y_test = df_test['is_canceled']
    
    # Guardar los archivos
    out_dir = './data/model_input/cv'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Guardando archivos en {out_dir}...")
    X_train_full.to_csv(os.path.join(out_dir, 'X_train_full.csv'), index=False)
    y_train_full.to_csv(os.path.join(out_dir, 'y_train_full.csv'), index=False)
    
    X_test.to_csv(os.path.join(out_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(out_dir, 'y_test.csv'), index=False)
    
    print("\n¡Preparación para Cross Validation completada!")
    print(f" - X_train_full (85%): {X_train_full.shape}")
    print(f" - X_test (15%):       {X_test.shape}")
    print("Nota: El objeto TimeSeriesSplit se instanciará directamente en el código de modelado.")

if __name__ == "__main__":
    main()


# Cargando datasets limpios...
# Generando matrices X e y para Cross Validation...
# Guardando archivos en ./data/model_input/cv...

# ¡Preparación para Cross Validation completada!
#  - X_train_full (85%): (74286, 23)
#  - X_test (15%):       (13110, 23)