"""Assignment 1 - training pipeline (step 1: split and cleaning)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def engineer_features(df):
    """Create new features and drop the columns they replace."""
    df = df.copy()
    # Age of the house and of the last remodel, measured at the time of sale
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    # Total living area and total bathrooms (half baths count 0.5)
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    # Yes/no indicators
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageType"] != "None").astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    # Drop the originals used above (and YrSold, see note)
    return df.drop(columns=["YrSold", "YearBuilt", "YearRemodAdd",
                            "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
                            "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"])

QUALITY_COLS = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


def encode_quality(df):
    """Turn the quality scales (None < Po < Fa < TA < Gd < Ex) into 0-5."""
    df = df.copy()
    for col in QUALITY_COLS:
        df[col] = df[col].map(QUALITY_MAP)
    return df

TARGET = "SalePrice"


def split_xy(df):
    """Separate the features (X) from the target (y)."""
    X = df.drop(columns=[TARGET]).copy()
    X["MSSubClass"] = X["MSSubClass"].astype(str)  # numeric code, but really a category
    return X, df[TARGET]


def fit_preprocessor(X_train):
    """Learn the scaling and one-hot encoding from TRAIN only."""
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return pre.fit(X_train)

ALPHAS = [1, 3, 10, 30, 100, 300]


def rmse(y_true, y_pred):
    """Root mean squared error, in dollars."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

if __name__ == "__main__":
    train, val, test = load_and_split()
    print("train / val / test:", len(train), len(val), len(test))
    stats = fit_cleaning(train)
    train, val, test = (apply_cleaning(d, stats) for d in (train, val, test))
    train, val, test = (engineer_features(d) for d in (train, val, test))
    train, val, test = (encode_quality(d) for d in (train, val, test))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = (split_xy(d) for d in (train, val, test))
    pre = fit_preprocessor(X_train)
    A_train, A_val, A_test = (pre.transform(X) for X in (X_train, X_val, X_test))

    # The target is standardised with TRAIN statistics; all three models learn on this scale
    y_mean, y_std = float(np.mean(y_train)), float(np.std(y_train))
    ys_train = (y_train.to_numpy() - y_mean) / y_std

    # Naive baseline: always predict the average training price
    print("baseline RMSE  val: %.0f  test: %.0f" % (rmse(y_val, y_mean), rmse(y_test, y_mean)))

    # scikit-learn Ridge: choose alpha on validation only
    scores = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(A_train, ys_train)
        scores[alpha] = rmse(y_val, model.predict(A_val) * y_std + y_mean)
        print("  alpha=%-4s val RMSE: %.0f" % (alpha, scores[alpha]))
    best_alpha = min(scores, key=scores.get)
    sk_model = Ridge(alpha=best_alpha).fit(A_train, ys_train)
    sk_test = rmse(y_test, sk_model.predict(A_test) * y_std + y_mean)
    print("scikit-learn Ridge (alpha=%s)  val: %.0f  test: %.0f" % (best_alpha, scores[best_alpha], sk_test))