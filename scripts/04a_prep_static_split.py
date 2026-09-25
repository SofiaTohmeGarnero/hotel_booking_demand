import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    print("Cargando datasets limpios...")
    processed_dir = './data/processed'
    train_path = os.path.join(processed_dir, 'train_clean.csv')
    test_path = os.path.join(processed_dir, 'test_clean.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("No se encontraron los archivos en data/processed/. Ejecuta 03_preprocessing.py primero.")
        
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print("Realizando split estático (70% Train, 15% Validation)...")
    # df_train_full representa el 85% original. Sacamos el 15% del total original para validación.
    # 0.15 / 0.85 = 0.17647
    df_train, df_val = train_test_split(df_train_full, test_size=(0.15/0.85), shuffle=False)
    
    # Columnas a eliminar para las matrices de features
    drop_para_modelo = ['is_canceled', 'booking_date', 'arrival_date']
    
    print("Generando matrices X e y...")
    X_train = df_train.drop(columns=drop_para_modelo)
    y_train = df_train['is_canceled']
    
    X_val = df_val.drop(columns=drop_para_modelo)
    y_val = df_val['is_canceled']
    
    X_test = df_test.drop(columns=drop_para_modelo)
    y_test = df_test['is_canceled']
    
    # Guardar los archivos
    out_dir = './data/model_input/static'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Guardando archivos en {out_dir}...")
    X_train.to_csv(os.path.join(out_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(out_dir, 'y_train.csv'), index=False)
    
    X_val.to_csv(os.path.join(out_dir, 'X_val.csv'), index=False)
    y_val.to_csv(os.path.join(out_dir, 'y_val.csv'), index=False)
    
    X_test.to_csv(os.path.join(out_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(out_dir, 'y_test.csv'), index=False)
    
    print("\n¡Preparación para Split Estático completada!")
    print(f" - X_train: {X_train.shape}")
    print(f" - X_val:   {X_val.shape}")
    print(f" - X_test:  {X_test.shape}")

if __name__ == "__main__":
    main()


# Cargando datasets limpios...
# Realizando split estático (70% Train, 15% Validation)...
# Generando matrices X e y...
# Guardando archivos en ./data/model_input/static...

# ¡Preparación para Split Estático completada!
#  - X_train: (61176, 23)
#  - X_val:   (13110, 23)
#  - X_test:  (13110, 23)