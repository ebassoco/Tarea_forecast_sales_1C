"""Pronostico de series tiempo

Este script:
    * Contiene las funciones necesarias para predecir h dias hacia
    adelante en una serie de tiempo utilizando un modelo jerarquico.

    * Puede ejecutarse directamente en la linea de comandos o se puede
    importar como modulo dentro de un programa principal.

Se requieren las liberias `statsforecast=2.0.0`, `hierarchicalforecast=1.0.1` y
`pandas=2.2.2` en el ambiente de trabajo.

El script recibe como insumos por omisión cuatro archivos en la ruta /model:
    * hts (StatsForecast): Objeto con el modelo entrenado.
    * df_series_jerarquicas (DataFrame): Dataframe con series de tiempo para
    todos los niveles de agregacion.
    * S_df (DataFrame): Dataframe con restricciones para la reconciliacion.
    * tags (Dictionary): Diccionario con las etiquetas de las series de tiempo
    de cada nivel jerarquico.
Ademas, predice automaticamente 30 dias.

El script escribe por omision DataFrame en la ruta /data/predictions.csv con
las predicciones de las series originales ajustadas por la reconciliacion
del tipo MinTrace.

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


# Importar paquetes
import argparse
from statsforecast.core import StatsForecast
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace
import pandas as pd



# Carga insumos de entrenamiento
def carga_insumos(path_insumos):
    '''Carga en memoria los insumos necesarios para el forecast

    Params:
        path_insumo (str): Ruta de los insumos.
    Return
        Una tupla con los siguientes 4 elementos:
            - hts (StatsForecast): Objeto con el modelo entrenado.
            - df_train (DataFrame): Dataframe con todas las series de tiempo
            - S_df (DataFrame): Matriz de restricciones
            - tags (Dictionary): Diccionario con las etiquetas de las series 
    '''

    hts = StatsForecast.load(f"{path_insumos}/hts")
    df_train = pd.read_pickle(f"{path_insumos}/df_hts.pickle")
    S_df = pd.read_pickle(f"{path_insumos}/S_df.pickle")
    tags = pd.read_pickle(f"{path_insumos}/tags.pickle")

    return hts, df_train, S_df, tags


# Pronosticos y reconciliacion
def pronostica_h_dias(model, df_train, h_pron):
    '''Genera dataframe con el pronostico de h_pron dias

    Params:
        model (StatsForecast): Objeto con el modelo entrenado.
        df_train (DataFrame): Dataframe con todas las series de tiempo.
        h_pron (int): Numero de dias a pronosticar.
    
    Return
        DataFrame con las predicciones de todas las series de tiempo. 
    '''
    return model.forecast(df=df_train, h=h_pron)

def reconciliacion_jerarquica(Y_hat, Y_train, S_df, tags):
    '''Genera dataframe con las series reconciliadas utilizando
    el metodo MinTrace.

    Params:
        Y_hat (DataFrame): Dataframe con las predicciones de las series
        de tiempo de todas las jerarquias.
        Y_train (DataFrame): Dataframe con las series de tiempo completas
        de entrenamiento.
        S_df (DataFrame): Matrix de restricciones.
        tags (Dictionary): Diccionario con las etiquetas de las series.
    
    Return
        DataFrame con la prediccion original y ajustada.
    '''

    reconcilers = [MinTrace(method='ols', nonnegative=True)]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)

    df_rec = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_train, S=S_df, tags=tags)

    return df_rec


# Formatos de salida de series originales
def filtra_series_originales(df, tags):
    '''Filtra dataframe con todas las series jerarquicas y conseva
    solo el nivel original sin agregacion

    Params:
        df (DataFrame): Dataframe con todas las series jerarquicas.
        tags (Dictionary): Diccionario con las etiquetas de las series.
    
    Return:
        DataFrame con las series originales.
    '''
    series_bottom = df.loc[
        lambda d: d["unique_id"].isin(
            tags['total/shop_id/item_id'])
    ]

    return series_bottom

def formatea_series_originales(df, niveles):
    '''Regresa dataframe con series reconciliadas a formato original
    de las variables de entrada

    Params:
        df (DataFrame): Dataframe con las series originales.
        niveles (List): Lista con los niveles de agregacion.
    
    Return:
        DataFrame con las series originales en formato original.
    '''

    df[niveles] = df['unique_id'].str.split('/', expand=True)
    df.drop('unique_id', axis=1, inplace=True)

    return df


# Función general del modulo
def genera_batch_pronostico(path_insumos = "/model",
                            h_pron = 30,
                            output_path = "/data"):
    '''Genera el pronostico h_pron dias hacia adelante para las
    series de tiempo originales con una agrupacion jerarquica.

    Params:
        path_insumos (str): Ruta de los insumos del entrenamiento.
        h_pron (int): Numero de dias a pronosticar.
        output_path (str): Ruta de salida de las predicciones.
    
    Return:
        Exporta un csv con las predicciones
    '''

    hts, df_train, S_df, tags = carga_insumos(path_insumos)

    Y_hat = pronostica_h_dias(hts, df_train, h_pron)
    df_rec = reconciliacion_jerarquica(Y_hat, df_train, S_df, tags)
    series_originales = filtra_series_originales(df_rec, tags)
    salida = formatea_series_originales(series_originales,
                                        ["total","shop_id", "item_id"])
    
    salida.to_csv(
        f"{output_path}/predictions_{pd.to_datetime('now').strftime('%Y%m%d_%H%M%S')}.csv",
        index=False)


# Para correr inference.py
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'path_insumo',
        type=str,
        help="Ruta de los insumos del entrenamiento"
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Ruta de los archivos de salida"
    )
    parser.add_argument(
        'h_pron',
        type=int,
        help="Estacionalidad en las series de tiempo"
    )
    args = parser.parse_args()

    pronostica_h_dias(path_insumos=args.path_insumo,
                      h_pron=args.h_pron,
                      output_path=args.output_path)

if __name__ == "__main__":
    main()
