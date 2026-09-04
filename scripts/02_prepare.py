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
    
    # 2. Separación Estratificada (Train 85%, Test 15%)
    # random_state=42 asegura que todos en el equipo obtengan el mismo corte
    print("Particionando los datos...")
    df_train_before_eda, df_test = train_test_split(
        df_clean, 
        test_size=0.15, 
        random_state=42, 
        stratify=df_clean['is_canceled']
    )
    
    # 3. Guardar los subconjuntos
    splits_dir = './data/splits'
    os.makedirs(splits_dir, exist_ok=True)
    
    df_train_before_eda.to_csv(os.path.join(splits_dir, 'train_before_eda.csv'), index=False)
    df_test.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
    
    print("¡Partición completada con éxito!")
    print(f" - Train: {df_train_before_eda.shape}")
    print(f" - Test: {df_test.shape}")

if __name__ == "__main__":
    main()

# Output
# Cargando el dataset original...
# Filas originales: 119390
# Filas sin duplicados: 87396
# Particionando los datos...
# ¡Partición completada con éxito!
#  - Train: (74286, 32)
#  - Test: (13110, 32)


# Nota: Eliminacion de duplicados

# * keep='first' (Por defecto): Mantiene la primera aparición de la fila y elimina los duplicados posteriores.
# >> df_clean = df.drop_duplicates()

# * keep='last': Mantiene la última aparición de la fila y elimina los duplicados anteriores. Útil si las filas de más abajo tienen información más actualizada.
# >> df_clean = df.drop_duplicates(keep='last')

# * keep=False: Elimina absolutamente todo lo que esté repetido. Si una fila aparece más de una vez, se borra por completo y no queda ni el original.
# >> df_clean = df.drop_duplicates(keep=False)