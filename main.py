# Librerias requeridas
# %pip install hierarchicalforecast=1.0.1 statsforecast=2.0.0 pandas=2.2.2
# %restart_python

# Importa modulos
from prep import prep_series_jeraquicas
from train import entrena_y_escribe_hts
from inference import genera_batch_pronostico


# Ejecuta un entrenamiento de 30 dias
prep_series_jeraquicas(
    file_path="/data/raw.csv",
    output_path="/data/prep.csv"
)
entrena_y_escribe_hts(
    path_insumo = "/data/prep.csv",
    jerarquias = ['total', 'shop_id', 'item_id'],
    season_len = 30,
    path_modelo = "/model"
)
genera_batch_pronostico(
    path_insumos="/model",
    h_pron=30,
    output_path="/data"
)

