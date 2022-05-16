

from src.data.extraction.constants import BIRTH_AGE, DEX, RADIOTHERAPY_DOSE
from src.data.extraction.data_management import PetaleDataManager
from src.data.processing.sampling import get_learning_one_data
from settings.paths import Paths
from torch import load
from os.path import join

DELETED_FEATURES = [BIRTH_AGE, DEX, RADIOTHERAPY_DOSE]

if __name__ == '__main__':

    # Data extraction
    df, target, cont_col, cat_col = get_learning_one_data(data_manager=PetaleDataManager(), genes=None)

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
    men_sex_coeff = params['_embedding_block._EntityEmbeddingBlock__embedding_layer.0.weight'][0, ]
    women_sex_coeff = params['_embedding_block._EntityEmbeddingBlock__embedding_layer.0.weight'][1, ]

    # Linear layer parameters
    coeff = params['_linear_layer.weight'].squeeze()

    # Sex coefficient in layer parameters
    men_sex_coeff = (men_sex_coeff*coeff[[5, 6]]).sum().item()
    women_sex_coeff = (women_sex_coeff*coeff[[5, 6]]).sum().item()

    # All biases
    bias = params['_linear_layer.bias'].item()
    biases = [women_sex_coeff + bias, men_sex_coeff + bias]

    # Coefficient corrections
    coeff = coeff.tolist()
    coeff = coeff[:5] + coeff[7:]

    for v, c in zip(variables, coeff + biases):
        print(f'{v} : {c:.2f}')