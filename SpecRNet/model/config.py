from typing import Dict

def feature_kwargs(lfcc: bool) -> dict:
    """Settings for preprocessing.
    """
    return {
        "n_lin": 128,
        "n_lfcc": 80,
        "log_lf": True,
        "speckwargs": {
            "n_fft": 512,
            "hop_length": 160,
            "win_length": 400
        },
    } if lfcc else {
        "n_mfcc": 20,
        "log_mels": True,
        "melkwargs": {
            "n_mels": 20,
            "n_fft": 512,
        }
    }

def get_specrnet_config(input_channels: int) -> Dict:
    return {
        "filts": [input_channels, [input_channels, 20], [20, 64], [64, 64]],
        "nb_fc_node": 64,
        "gru_node": 64,
        "nb_gru_layer": 2,
        "nb_classes": 1,
    }
