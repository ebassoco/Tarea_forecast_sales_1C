"""Preparacion de datos de entrenamiento

Este script:
    * Contiene las funciones necesarias para leer el archivo con los 
    datos de series de tiempo de ventas, procesarlas y construir el
    dataframe con el formato requerido por las librerias statsforecast
    y hierarchicalforecast.

    * Puede ejecutarse directamente en la linea de comandos o se puede
    importar como modulo dentro de un programa principal.

La heramienta acepta archivos en formato csv. Además requiere que la 
liberia `pandas` este instalada en el entorno de ejecucion.

En caso de que no se defina un nombre de archivo, se busca el archivo 
`raw.csv` en el directorio /data. En caso que no se defina una ruta de
salida, el archivo se guarda en el directorio /data con el nombre prep.csv.

Estas son las funciones que contiene:
    * carga_df_ventas: Lee el archivo csv con las ventas diarias y lo guarda 
    como dataframe.
    * define_columna_tiempo: agrega la columna "ds" en formato datetime.
    * encuentra_series_con_n_periodos: Encuentra series con un numero
    especifico de periodos de ventas mayores a 0.
    * filtra_series_completas: Filtra el dataframe con las series completas,
    es decir, las que tienen n periodos de ventas mayores a 0.
    * seleccion_variables_hts: Selecciona las variables necesarias para el 
    entrenamiento usando la librería hierarchicalforecast.
    * completa_ceros_en_series: Completa los gaps en cada una de las series 
    diarias.
    * agrega_columna_total: Agrega columna con nivel total de agregación.
    * prep_series_jeraquicas: Funcion principal del script para su ejecucion
    en otro script.
    * main: Funcion principal de script para su ejecucion en linea de comandos.
"""



# Importar paquetes
import argparse
import pandas as pd



# Lectura de datos
def carga_df_ventas(file_path):
    '''Lectura del archivo csv con las ventas diarias por item_id/shop_id.

    Params:
        file_path (str): Ruta del archivo de ventas

    Returns:
        Un dataframe con las ventas diarias por item_id/shop_id.
    '''
    # Lectura del csv

    return pd.read_csv(file_path)



# Preprocesamiento de datos
def define_columna_tiempo(df, var_time = "date"):
    '''Creación de la variable "ds" en formato datetime.

    Params:
        df (DataFrame): Dataframe con variable var_time en formato string
        var_time (str): Nombre de la columna con la variable de tiempo

    Returns:
        Un dataframe con variable ds (datetime).
    '''

    df["ds"] = pd.to_datetime(df[var_time], format="%d.%m.%Y")
    df = df.drop(["date"], axis=1)

    return df

def encuentra_series_con_n_periodos(df,
                                    var_ids = ["shop_id", "item_id"],
                                    var_bloq = "date_block_num",
                                    n_peridos=34):
    '''Obtención de índices de series con n_peridos periodos de ventas > 0.

    Params:
        df (DataFrame): Dataframe con variable var_bloq con periodos de tiempo.
        var_ids (list): Lista de variables identificadoras de series de tiempo.
        var_bloq (str): Nombre de la columna con la variable de periodos de tiempo.
        n_periodos (int): Número de periodos con ventas > 0.

    Returns:
        Un dataframe con series con ventas>0 en n_periodos.
    '''

    ## Series completas (n_peridos con ventas>0)
    df = df.groupby(
        var_ids
    ).agg(
        {var_bloq: "nunique"}
    ).loc[
        lambda d: d[var_bloq] == n_peridos
    ].drop(
        [var_bloq], axis=1
    )

    return df

def filtra_series_completas(df_original, idx_series_completas):
    '''Filtro de series de tiempo completas.

    Params:
        df_original (DataFrame): Dataframe original con todas las series de tiempo.
        idx_series_completas (DataFrame): Dataframe con identificadores de series completas.

    Returns:
        Un dataframe con series completas (con ventas>0 en todos los periodos).
    '''


    return df_original.join(idx_series_completas,
                            on=["shop_id", "item_id"],
                            how="inner")
    


# Prepocesamiento series para modelo HTS
def seleccion_variables_hts(df,
                            var_ids = ["shop_id", "item_id", "ds", "item_cnt_day"],
                            var_time = "ds",
                            var_sales = "item_cnt_day"):
    '''Selección de variables necesarias para entrenamiento de modelos HTS.

    Params:
        df (DataFrame): Dataframe con series de tiempo.
        var_ids (list): Lista de variables con identificadores de series de tiempo.
        var_time (str): Nombre de la columna con la variable de periodos de tiempo.
        var_sales (str): Nombre de la columna con la variable de ventas.

    Returns:
        Un dataframe con columnas necesarias para entrenamiento.
    '''
    
    if len(var_ids) == 4:
        vars = var_ids
    else:
        vars = var_ids.extend(var_time).extend(var_sales)
    

    df = df[vars]

    df["y"] = df[var_sales]
    df.drop([var_sales], axis=1, inplace=True)

    return df

def completa_ceros_en_series(df):
    '''Construye DF con todas las combinaciones de fecha/shop-item.

    Params:
        df (DataFrame): Dataframe con series de tiempo y variables "ds", "shop_id" e "item_id".

    Returns:
        Un dataframe con todas las combinaciones (ceros completados).
    '''

    # Valores de fechas min, max
    min_date = df.ds.min()
    max_date = df.ds.max()
    rango_fechas = pd.date_range(min_date, max_date)

    # Combinaciones fecha-tienda-item
    idx = pd.MultiIndex.from_product(
        [rango_fechas, df.shop_id.unique(), df.item_id.unique()],
        names=['ds', 'shop_id', 'item_id'])
    
    # Combinaciones tienda-item
    idx_ids = df[['shop_id', 'item_id']].drop_duplicates().set_index(['shop_id','item_id'])

    df = df.set_index(
        ["ds", "shop_id", "item_id"]
    ).reindex(idx).fillna(0).reset_index().drop(
        'date_block_num',axis=1
    ).join(idx_ids, on=['shop_id', 'item_id'], how='inner')

    return df

def agrega_columna_total(df, nom_total="Total"):
    '''Agrega columna con nivel total de agregación.

    Params:
        df (DataFrame): Dataframe con series de tiempo.

    Returns:
        Un dataframe con columna para nivel total de agregación.
    '''

    df['total'] = nom_total

    return df

# Función general del modulo
def prep_series_jeraquicas(file_path="/data/raw.csv",
                           output_path="/data/prep.csv"):
    '''Genera un archivo csv con el dataframe en el formato
    necesario para el entrenamiento de series de tiempo jerarquicas.

    Params:
        file_path (str): Ruta del archivo de ventas.
        output_path (str): Ruta del archivo de salida.

    Returns:
        Exporta un csv en la ruta definida
    '''

    sales = carga_df_ventas(file_path) 
    sales = define_columna_tiempo(sales)

    series_completas = filtra_series_completas(
        sales,
        encuentra_series_con_n_periodos(sales)) 

    df_modelo = agrega_columna_total(
        seleccion_variables_hts(
            completa_ceros_en_series(series_completas)
            )
        )
    df_modelo.to_csv(output_path, index=False)


# Para correr prep.py
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_file',
        type=str,
        help="Archivo con las ventas por dia/tienda/item"
    )
    parser.add_argument(
        'output_file',
        type=str,
        help="Ruta/nombre del archivo de salida."
    )
    args = parser.parse_args()

    prep_series_jeraquicas(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
