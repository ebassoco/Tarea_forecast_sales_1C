"""Entrenamiento de series de tiempo jerarquicas

Este script:
    * Contiene las funciones necesarias para el entrenamiento de un
    modelo de series de tiempo jerarquicas.

    * Puede ejecutarse directamente en la linea de comandos o se puede
    importar como modulo dentro de un programa principal.

Se requieren las liberias `statsforecast=2.0.0`, `hierarchicalforecast=1.0.1` y
`pandas=2.2.2` en el ambiente de trabajo.

El script recibe como insumos por omisión un dataframe con las series
de tiempo por dia que se quieren pronosticar en el archivo /data/prep.csv.
En otro caso, se debe especificar la ruta del archivo csv con las series
con las columnas: "ds" (datetime) y "y" (float).

El script escribe por omision un objeto de la clase
statsforecast.core.StatsForecast y tres archivos .pkl en la ruta /model.


Estas son las funciones que contiene:
    * define_jerarquia(niveles): Define la jerarquía de las series.
    * genera_insumos_agregacion(df, jeraquias): Genera los insumos necesarios
    para el entrenamiento de series de tiempo jerarquicas: dataframe con 
    todas las series (finales y agregadas), matriz de restricciones y 
    diccionario con etiquetas de las series en cada nivel jerarquico
    * entrenamiento_autoets(season_len): Define el objeto StatsForecast con el
    el modelo AutoETS.
    * entrena_y_escribe_hts(path_insumo, jerarquias, season_len, path_modelo):
    funcion principal del script para su ejecucion en otro scritp
    * main(): Función principal para su ejecución en linea de comandos.
"""


#train.py

# Importa paquetes
import argparse
from statsforecast.core import StatsForecast
from statsforecast.models import AutoETS
from hierarchicalforecast.utils import aggregate
import pandas as pd


# Funciones auxiliares preentrenamiento
def define_jerarquia(niveles):
    '''Regresa arreglo con los niveles de jerarquía.

    Params:
        niveles (list): Lista de columnas que definen la jerarquía en orden descendente.

    Returns:
        Un arreglo con las jerarquías
    '''
    jeraquias = []

    for i in range(len(niveles)):
        jeraquias.insert(i, niveles[:(i+1)])
    return jeraquias

def genera_insumos_agregacion(df, jeraquias):
    '''Regresa una tupla con los elementos necesarios para el
    entrenamiento de series de tiempo jerarquicas.

    Params:
        df (DataFrame): DataFrame con variables ds(datetime), y(double) y 
        columnas (str) para cada nivel de agregacion jerarquico.
        jerarquias (list): Lista de columnas que definen la jerarquía
        en orden descendente.

    Returns:
        df_jerarquico (DataFrame): DataFrame con las series de tiempo
        agregadas y columnas "unique_id", "ds" y "y".
        S_df (DataFrame): DataFrame con la matriz de restricciones de tamaño
        nxm con n= numero de series contando agregaciones y m= series originales.
        tags (dictionary): Diccionario con las etiquetas de las series de cada nivel
        jerarquico.
    '''
    
    df_jerarquico, S_df, tags = aggregate(df=df, spec=jeraquias)

    return df_jerarquico, S_df, tags


# Entrenamiento AutoETS
def entrenamiento_autoets(season_len):
    '''Entrena un modelo AutoETS a un conjunto de series de tiempo
    diarias.

    Params:
        df (DataFrame): DataFrame con variables unique_id, ds, y.
        season_len (int): Estacionalidad de la serie de tiempo.
    Returns:
        model (StatsForecast): Modelo entrenado.
    '''
    
    model = StatsForecast(
        models=[AutoETS(season_length=season_len)],
        freq='D',
        n_jobs=-1)
    
    return model


# Función general del modulo
def entrena_y_escribe_hts(path_insumo = "data/prep.csv",
                          jerarquias = ['total', 'shop_id', 'item_id'],
                          season_len = 30,
                          path_modelo = "model/"):
    '''Entrena un modelo AutoETS a un conjunto de series de tiempo
    diarias que lee a partir de un archivo csv y después lo exporta.

    Params:
        path_insumo (str): Ruta del archivo csv con las series de tiempo y
        las columnas "unique_id", "ds", "y".
        jerarquias (list): Lista de columnas que definen la jerarquía en 
        orden descendente.
        season_len (int): Estacionalidad de las serie de tiempo.
        path_modelo (str): Ruta donde se escribe el modelo y los insumos
        relacionados.
    Return
        Exporta en path_modelo:
            hts (StatsForecast): Modelo entrenado
            df_hts (DataFrame): DataFrame con las series de tiempo jerarquicas.
            S_df (DataFrame): DataFrame con la matriz de restricciones.
            tags (dictionary): Diccionario con las etiquetas de las
            series de cada nivel jerarquico.

    '''

    jerarqs = define_jerarquia(jerarquias)

    df_train = pd.read_csv(path_insumo)

    df_jeraquico, S_df, tags = genera_insumos_agregacion(df_train, jerarqs)

    model = entrenamiento_autoets(season_len)

    model.save(f"{path_modelo}/hts")
    df_jeraquico.to_pickle(f"{path_modelo}/df_hts.pickle")
    S_df.to_pickle(f"{path_modelo}/S_df.pickle")
    pd.to_pickle(tags, f"{path_modelo}/tags.pickle")
                              
# Para correr train.py
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_file',
        type=str,
        help="Archivo preprocesado con columnas unique_id, ds y y"
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Ruta de los archivos de salida"
    )
    parser.add_argument(
        'season_length',
        type=int,
        help="Estacionalidad en las series de tiempo"
    )
    args = parser.parse_args()

    entrena_y_escribe_hts(path_insumo = args.input_file,
                          jerarquias = ['total', 'shop_id', 'item_id'],
                          season_len = args.season_length,
                          path_modelo = args.output_path)

if __name__ == "__main__":
    main()
