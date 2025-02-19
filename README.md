# Pronóstico de Ventas para Tiendas 1C

Este proyecto realiza un pronóstico de ventas de las tiendas 1C en Rusia utilizando series de tiempo jerárquicas.


## Datos
Los datos provienen de [Kaggle: Predict Future Sales](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data), 
los cuales cubren las ventas diarias desde enero de 2013 hasta octubre de 2015.

Sólamente se utilizó el archivo `sales_train.csv` del cual conservamos las columnas:
- shop_id
- item_id
- date
- date_block_num
- item_cnt_day


## Pasos del Proyecto
El objetivo principal de este proyecto es predecir las ventas futuras de las tiendas utilizando un modelo de series de tiempo jerárquicas.
Para ello, el proceso se divide en tres módulos:

- **`prep.py`**: Este módulo realiza la carga del insumo original y exporta un csv con las columnas en el formato adecuado para el ajuste del modelo.
  + El script lee el archivo raw.csv del directorio /data el cual es el mismo que descargamos de Kaggle. La ruta puede modificarse.
  + Se filtran las series de tiempo 'completas', es decir, aquellas que tienen información de ventas en al menos n=34 meses. Este parámetro puede modificarse dentro del script.
  + Se completan ceros (días sin registros de ventas) en las series diarias que cumplieron el crieterio de ventas en 34 meses.
  + Se preparan las columnas con el nombre que espera la librería statsforecast: "ds" para el índice de tiempo y "y" para la variable objetivo.
  + Se exporta un archivo prep.csv con las series diarias completas en la carpeta /data. La ruta puede modificarse.
- **`train.py`**: Este módulo construye las series de tiempo agregadas y luego ajusta un modelo de suavizado exponencial (ETS) a cada una.
  + El script recibe como insumo el archivo /data/prep.csv que procesamos en el paso anterior.
  + Se deben definir las jerarquias en las series de tiempo, para este proyecto se definieron como: Total | Tienda | Item. El modelo jerárquico ajusta un ETS a cada nivel de agregación y despúes reconcilia los datos para garantizar la coherencia en los pronósticos en todos los niveles jerárquicos.
  + Se exportan 4 archivos en la carpeta /model: *hts* que contiene el objeto StatsForecast con los ETS ajustados, *df_hts.pickle* con las series de tiempo de todos los niveles de agregación, *S_df.pickle* la matriz de restricciones y *tags.pickle* el detalle de las etiquetas en cada nivel jerarquico.
- **`inference.py`**: Realiza el pronóstico de ventas a futuro e imprime los resultados en un archivo CSV.
  + Se leen los 4 insumos del paso anterior y se utilizan para el pronóstico de h días hacia adelante
  + Por omisión, el pronóstico se hace para 30 días aunque puede cambiarse este horizonte.
  + Se exporta el archivo /data/prediccion_fecha_hora.csv con los pronósticos para las series originales.


## Requisitos
Este proyecto requiere las siguientes librerías:

- `pandas` (versión 2.2.2 o superior)
- `statsforecast` (version 2.0.0 para el ajuste del modelo ETS)
- `hierarchicalforecast` (version 1.0.0 para la reconciliación jerárquica)


## Ejecución
Para la ejecución de este proyecto, se deben seguir los siguientes pasos:

1. Clonar este repositorio o hacer a mano un directorio que contenga la misma estructura (de carpetas y archivos)
2. Verificar que se tengan las dependencias instaladas, de no ser así, descomentar las primeras dós lineas de código del script main.py
3. Ejecutar el script main.py

Si se desean cambiar las rutas/nombres de insumos o el horizonte de pronóstico, se modifican los valores del script main.py



## Estructura del repositorio

```plaintext
.
Tarea_forecast_sales_1C/
├── data/                          # Archivos de datos
│   ├── raw.csv                    # Datos originales
│   ├── prep.csv                   # Datos procesados
│   └── prediction_fecha_hora.csv  # Resultados de predicción
├── model/                         # Archivos de modelo entrenado
│   ├── hts                        # Modelo entrenado
│   ├── df_hts.pickle              # Series jerarquicas en formato pickle
│   ├── S_df.pickle                # Matriz de restricciones en formato pickle
│   └── tags.pickle                # Tags de etiquetas en formato pickle
├── src/                           # Archivos de código fuente
│   ├── prep.py                    # Preprocesamiento de datos
│   ├── train.py                   # Entrenamiento del modelo ETS
│   ├── inference.py               # Pronóstico y resultados
│   └── __init__.py                
├── main.py                        # Script principal para ejecutar el flujo


