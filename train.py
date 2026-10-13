"""Assignment 1 - training pipeline (step 1: split and cleaning)."""
import pandas as pd

DATA_PATH = "data/ames.csv"

# Columns where an empty value means "the house does not have this feature"
NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]


def load_and_split():
    """Temporal split: the model is trained on the past and tested on the future."""
    df = pd.read_csv(DATA_PATH).drop(columns=["Id"])
    train = df[df["YrSold"] <= 2008].copy()
    val = df[df["YrSold"] == 2009].copy()
    test = df[df["YrSold"] == 2010].copy()
    return train, val, test


def fit_cleaning(train):
    """Learn the fill values using ONLY the training data (avoids leakage)."""
    return {
        "lot_by_neighborhood": train.groupby("Neighborhood")["LotFrontage"].median(),
        "lot_global": train["LotFrontage"].median(),
        "electrical_mode": train["Electrical"].mode()[0],
    }


def apply_cleaning(df, stats):
    """Apply the fill values learned on train to any split."""
    df = df.copy()
    df[NONE_COLS] = df[NONE_COLS].fillna("None")
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["Electrical"] = df["Electrical"].fillna(stats["electrical_mode"])
    lot_fill = df["Neighborhood"].map(stats["lot_by_neighborhood"]).fillna(stats["lot_global"])
    df["LotFrontage"] = df["LotFrontage"].fillna(lot_fill)
    df = df.drop(columns=["GarageYrBlt"])
    return df


if __name__ == "__main__":
    train, val, test = load_and_split()
    print("train / val / test:", len(train), len(val), len(test))
    stats = fit_cleaning(train)
    train, val, test = (apply_cleaning(d, stats) for d in (train, val, test))
    for name, d in (("train", train), ("val", val), ("test", test)):
        na = d.isna().sum()
        print(name, "remaining missing:", na[na > 0].to_dict())
