import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    print("Cargando el dataset original...")
    # Leer el archivo que generó el script 01
    df = pd.read_csv('./data/raw/hotel_bookings.csv')
    print(f"Filas originales: {df.shape[0]}")
    
    # 1. Eliminar duplicados
    df_clean = df.drop_duplicates()
    print(f"Filas sin duplicados: {df_clean.shape[0]}")
    
    # 2. Separación Estratificada (Train 70%, Val 15%, Test 15%)
    # random_state=42 asegura que todos en el equipo obtengan el mismo corte
    print("Particionando los datos...")
    df_train, df_temp = train_test_split(
        df_clean, 
        test_size=0.30, 
        random_state=42, 
        stratify=df_clean['is_canceled']
    )
    
    df_val, df_test = train_test_split(
        df_temp, 
        test_size=0.50, 
        random_state=42, 
        stratify=df_temp['is_canceled']
    )
    
    # 3. Guardar los subconjuntos
    splits_dir = './data/splits'
    os.makedirs(splits_dir, exist_ok=True)
    
    df_train.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
    
    print("¡Partición completada con éxito!")
    print(f" - Train: {df_train.shape}")
    print(f" - Validation: {df_val.shape}")
    print(f" - Test: {df_test.shape}")

if __name__ == "__main__":
    main()

# Output
# Cargando el dataset original...
# Filas originales: 119390
# Filas sin duplicados: 87396
# Particionando los datos...
# ¡Partición completada con éxito!
#  - Train: (61177, 32) ---> (filas, columnas)
#  - Validation: (13109, 32)
#  - Test: (13110, 32)