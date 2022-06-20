import matplotlib.pyplot as plt
from pandas import merge
from src.data.extraction.constants import DATE, DATE_OF_TREATMENT_END, GEN_2, PARTICIPANT, PHASE, TAG, TSEOT
from src.data.extraction.data_management import PetaleDataManager
from src.data.extraction.helpers import get_abs_years_timelapse
from src.data.processing.sampling import get_learning_one_data, get_warmup_data

if __name__ == '__main__':

    # Data manager creation
    m = PetaleDataManager()

    # Years delta data extraction
    time_df = m.get_table(GEN_2, columns=[PARTICIPANT, TAG, DATE, DATE_OF_TREATMENT_END])
    time_df = time_df[time_df[TAG] == PHASE]
    get_abs_years_timelapse(time_df, TSEOT, DATE_OF_TREATMENT_END, DATE)
    time_df.drop([DATE, DATE_OF_TREATMENT_END, TAG], axis=1, inplace=True)

    """
    VO2 timeline
    """
    # VO2 data extraction
    df, _, _, _ = get_warmup_data(m, baselines=True, genes=None, holdout=True)

    # Data merge
    df = merge(df, time_df, on=PARTICIPANT)
    plt.hist(df[TSEOT].to_numpy(dtype=float), bins=5)
    plt.show()

    # Data analysis
    print("VO2 timeline stats")
    print(f"Mean : {df[TSEOT].mean():.2f}")
    print(f"Std : {df[TSEOT].std():.2f}")
    print(f"Min : {df[TSEOT].min():.2f}")
    print(f"Max : {df[TSEOT].max():.2f}")
    print(f"Median : {df[TSEOT].median():.2f}")

    """
    Obesity timeline
    """
    # Obesity data extraction
    df, _, _, _ = get_learning_one_data(m, genes=None, baselines=False, holdout=True)

    # Data merge
    df = merge(df, time_df, on=PARTICIPANT)
    plt.hist(df[TSEOT].to_numpy(dtype=float), bins=5)
    plt.show()

    # Data analysis
    print("\n\nObesity timeline stats")
    print(f"Mean : {df[TSEOT].mean():.2f}")
    print(f"Std : {df[TSEOT].std():.2f}")
    print(f"Min : {df[TSEOT].min():.2f}")
    print(f"Max : {df[TSEOT].max():.2f}")
    print(f"Median : {df[TSEOT].median():.2f}")