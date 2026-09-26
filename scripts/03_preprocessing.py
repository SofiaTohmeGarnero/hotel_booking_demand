import os
import pandas as pd

def aplicar_limpieza(df: pd.DataFrame, adr_median: float) -> pd.DataFrame:
    """
    Aplica las reglas de limpieza y feature engineering definidas en el EDA.
    """
    df_clean = df.copy()
    
    # 1. Eliminación de Data Leakage
    cols_to_drop = [
        'reservation_status', 'reservation_status_date', 'assigned_room_type', 
        'booking_changes', 'arrival_date_day_of_month', 'arrival_date_week_number',
        'previous_bookings_not_canceled', 'arrival_date_month', 'arrival_date_str'
    ]
    df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], inplace=True)
    
    # 2. Nulos
    df_clean['children'] = df_clean['children'].fillna(0)
    df_clean['country'] = df_clean['country'].fillna('Unknown')
    df_clean['agent'] = df_clean['agent'].fillna(0)
    df_clean['company'] = df_clean['company'].fillna(0)
    
    # 3. Outliers
    df_clean.loc[df_clean['adr'] < 0, 'adr'] = 0.0
    df_clean.loc[df_clean['adr'] > 400, 'adr'] = adr_median
    df_clean.loc[(df_clean['babies'] > 3) & (df_clean['adults'] <= 2), 'babies'] = 1
    
    # 4. Feature Engineering
    df_clean['total_nights'] = df_clean['stays_in_week_nights'] + df_clean['stays_in_weekend_nights']
    df_clean.drop(columns=['stays_in_week_nights', 'stays_in_weekend_nights'], inplace=True)
    
    df_clean['is_placed_on_waiting_list'] = (df_clean['days_in_waiting_list'] > 0).astype(int)
    df_clean.drop(columns=['days_in_waiting_list'], inplace=True)
    
    df_clean['is_portugal'] = (df_clean['country'] == 'PRT').astype(int)
    df_clean.drop(columns=['country'], inplace=True)
    df_clean['is_resort'] = (df_clean['hotel'] == 'Resort Hotel').astype(int)
    df_clean.drop(columns=['hotel'], inplace=True)
    
    
    return df_clean

def main():
    print("Cargando los datasets particionados...")
    splits_dir = './data/time-based-splits'
    train_path = os.path.join(splits_dir, 'train_before_eda.csv')
    test_path = os.path.join(splits_dir, 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("No se encontraron los archivos en data/time-based-splits/. Ejecuta 02_time-based-splitting.py primero.")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Dimensiones iniciales - Train: {df_train.shape}, Test: {df_test.shape}")
    
    # Calcular la mediana de ADR SOLO en el set de Train
    print("Calculando estadísticos en Train (adr median)...")
    adr_median_train = df_train['adr'].median()
    print(f"Mediana de ADR calculada en Train: {adr_median_train}")
    
    # Aplicar limpieza
    print("Aplicando limpieza y feature engineering a Train y Test...")
    df_train_clean = aplicar_limpieza(df_train, adr_median_train)
    df_test_clean = aplicar_limpieza(df_test, adr_median_train)
    
    print(f"Dimensiones finales - Train: {df_train_clean.shape}, Test: {df_test_clean.shape}")
    
    # Guardar los datasets procesados
    processed_dir = './data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    train_out_path = os.path.join(processed_dir, 'train_clean.csv')
    test_out_path = os.path.join(processed_dir, 'test_clean.csv')
    
    df_train_clean.to_csv(train_out_path, index=False)
    df_test_clean.to_csv(test_out_path, index=False)
    
    print(f"\n¡Preprocesamiento completado! Archivos guardados en {processed_dir}")

if __name__ == "__main__":
    main()


# Cargando los datasets particionados...
# Dimensiones iniciales - Train: (74286, 35), Test: (13110, 35)
# Calculando estadísticos en Train (adr median)...
# Mediana de ADR calculada en Train: 93.6
# Aplicando limpieza y feature engineering a Train y Test...
# Dimensiones finales - Train: (74286, 26), Test: (13110, 26)