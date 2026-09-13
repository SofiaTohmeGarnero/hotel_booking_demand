import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_booking_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la variable temporal 'booking_date' a partir de los datos de llegada 
    y el 'lead_time' (arrival_date - lead_time).
    """
    df_copy = df.copy()
    
    # Mapeo de meses a formato numérico
    months_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    # Construir fecha de llegada completa
    df_copy['arrival_month_num'] = df_copy['arrival_date_month'].map(months_map)
    df_copy['arrival_date'] = pd.to_datetime(
        df_copy['arrival_date_year'].astype(str) + '-' + 
        df_copy['arrival_month_num'].astype(str).str.zfill(2) + '-' + 
        df_copy['arrival_date_day_of_month'].astype(str).str.zfill(2)
    )
    
    # Calcular fecha en la que se realizó la reserva
    df_copy['booking_date'] = df_copy['arrival_date'] - pd.to_timedelta(df_copy['lead_time'], unit='D')
    
    # Eliminar columnas auxiliares para mantener limpio el DataFrame
    df_copy.drop(columns=['arrival_month_num', 'arrival_date'], inplace=True)
    
    return df_copy


def main():
    print("Cargando el dataset original...")
    raw_path = './data/raw/hotel_bookings.csv'
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encontró el archivo en {raw_path}")
        
    df = pd.read_csv(raw_path)
    print(f"Filas originales: {df.shape[0]}")
    
    # 1. Eliminar duplicados
    df_clean = df.drop_duplicates().copy()
    print(f"Filas sin duplicados: {df_clean.shape[0]}")
    
    # 2. Generar la variable booking_date
    print("Calculando la variable 'booking_date'...")
    df_clean = create_booking_date(df_clean)
    
    # 3. Ordenar el dataset cronológicamente por la fecha de reserva
    df_sorted = df_clean.sort_values(by='booking_date').reset_index(drop=True)
    
    # 4. Separación Temporal (Train 85%, Test 15%)
    
    # test_ratio = 0.15
    # split_index = int(len(df_sorted) * (1 - test_ratio))
    
    print("Particionando los datos cronológicamente...")
    # df_train_before_eda = df_sorted.iloc[:split_index].copy()
    # df_test = df_sorted.iloc[split_index:].copy()
    df_train_before_eda, df_test = train_test_split(
      df_sorted, 
      test_size=0.15, 
      shuffle=False
    )
    
    # 5. Guardar los subconjuntos
    splits_dir = './data/time-based-splits'
    os.makedirs(splits_dir, exist_ok=True)
    
    train_out_path = os.path.join(splits_dir, 'train_before_eda.csv')
    test_out_path = os.path.join(splits_dir, 'test.csv')
    
    df_train_before_eda.to_csv(train_out_path, index=False)
    df_test.to_csv(test_out_path, index=False)
    
    # 6. Reporte de la partición
    print("\n¡Partición temporal completada con éxito!")
    print(f" - Train Shape: {df_train_before_eda.shape}")
    print(f"   * Rango booking_date Train: {df_train_before_eda['booking_date'].min().date()} a {df_train_before_eda['booking_date'].max().date()}")
    print(f"   * Tasa de cancelación Train: {df_train_before_eda['is_canceled'].mean():.2%}")
    
    print(f" - Test Shape:  {df_test.shape}")
    print(f"   * Rango booking_date Test:  {df_test['booking_date'].min().date()} a {df_test['booking_date'].max().date()}")
    print(f"   * Tasa de cancelación Test:  {df_test['is_canceled'].mean():.2%}")

if __name__ == "__main__":
    main()


# Output

# Cargando el dataset original...
# Filas originales: 119390
# Filas sin duplicados: 87396
# Calculando la variable 'booking_date'...
# Particionando los datos cronológicamente...

# ¡Partición temporal completada con éxito!
#  - Train Shape: (74286, 33)
#    * Rango booking_date Train: 2013-06-24 a 2017-02-21
#    * Tasa de cancelación Train: 27.54%
#  - Test Shape:  (13110, 33)
#    * Rango booking_date Test:  2017-02-21 a 2017-08-31
#    * Tasa de cancelación Test:  27.22%