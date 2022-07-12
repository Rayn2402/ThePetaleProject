"""
Filename: coefficients.py

Author: Nicolas Raymond

Description: Script used to coefficients of the last linear layer
             of the total body fat prediction model

Date of last modification: 2022/07/12
"""

import sys
from os.path import dirname, join, realpath
from torch import load

if __name__ == '__main__':

    # Imports specific to project
    sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))
    from src.data.extraction.constants import BIRTH_AGE, DEX, RADIOTHERAPY_DOSE
    from src.data.extraction.data_management import PetaleDataManager
    from src.data.processing.sampling import get_obesity_data
    from settings.paths import Paths

    # Deleted features initialization
    DELETED_FEATURES = [BIRTH_AGE, DEX, RADIOTHERAPY_DOSE]

    # Data extraction
    _, _, cont_col, cat_col = get_obesity_data(PetaleDataManager(), genomics=False)

    # Feature filtering
    for c in DELETED_FEATURES:
        if c in cont_col:
            cont_col.remove(c)
        else:
            cat_col.remove(c)

    # Variables list initialization
    variables = cont_col + [f's{i}' for i in range(4)] + ['sex_bias_w', 'sex_bias_m']

    # Parameter loading
    params = load(join(Paths.MODELS, 'obesity_ggae.pt'))

    # Sex categorical embedding
    men_sex_coefficient = params['_embedding_block._EntityEmbeddingBlock__embedding_layer.0.weight'][0, ]
    women_sex_coefficient = params['_embedding_block._EntityEmbeddingBlock__embedding_layer.0.weight'][1, ]

    # Linear layer parameters
    coefficients = params['_linear_layer.weight'].squeeze()

    # Sex coefficient in layer parameters
    men_sex_coefficient = (men_sex_coefficient*coefficients[[6, 7]]).sum().item()
    women_sex_coefficient = (women_sex_coefficient*coefficients[[6, 7]]).sum().item()

    # All biases
    bias = params['_linear_layer.bias'].item()
    biases = [women_sex_coefficient + bias, men_sex_coefficient + bias]

    # Coefficient corrections
    coefficients = coefficients.tolist()
    coefficients = coefficients[:6] + coefficients[8:]

    for v, c in zip(variables, coefficients + biases):
        print(f'{v} : {c:.2f}')