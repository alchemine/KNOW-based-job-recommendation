from analysis_tools.common import *
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.binary import BinaryEncoder
from sklearn.cluster import OPTICS


def dtype_convertor(df):
    for col in df:
        try:
            df[col] = df[col].astype(np.int32)
        except:
            df[col] = df[col].astype('category')
    return df


class DFTransformer:
    def __init__(self, processor):
        self.processor = processor
    def fit(self, X, y=None):
        self.processor.fit(X, y)
        return self
    def transform(self, X):
        return pd.DataFrame(self.processor.transform(X), index=X.index)


def get_simple_imputer(col_regex, missing_values, fill_value):
    def imputer(df):
        cols     = df.filter(regex=col_regex).columns
        df[cols] = df[cols].replace(missing_values, fill_value)
        return df
    return imputer


def get_preprocessor_baseline():
    return make_pipeline(
        DFTransformer(SimpleImputer(missing_values=' ', strategy="constant", fill_value='0')),
        OrdinalEncoder()
    )

def get_imputer():
    return make_pipeline(
        FunctionTransformer(lambda df: df.replace(["^없.*$", "^.*모름.*$", "^.*모르겠음.*$"], ['모름', '모름', '모름'], regex=True)),
        FunctionTransformer(lambda df: df.replace(['PC', '^선생님$'], ['컴퓨터', '교사'], regex=True)),

        FunctionTransformer(get_simple_imputer("^aq.*_2$", ' ', '0')),
        FunctionTransformer(get_simple_imputer("^bq5_1$", ' ', '0')),
        FunctionTransformer(get_simple_imputer("^bq12_[234]$", ' ', '3')),
        FunctionTransformer(get_simple_imputer("^bq40$", ' ', '0')),
        FunctionTransformer(get_simple_imputer("^bq41_.*$", ' ', '0')),
        FunctionTransformer(get_simple_imputer("^bq.*$", ' ', '모름')),

        FunctionTransformer(dtype_convertor),
    )

def get_preprocessor1():
    return make_pipeline(
        get_imputer(),
        DFTransformer(make_column_transformer(
            (BinaryEncoder(), make_column_selector(dtype_include='category')),
            remainder='passthrough'
        ))
    )


def preprocess2_y(X, y):
    idxs_unknown_label = y[y == 9999999].index
    X_unknown = X.loc[idxs_unknown_label]  # deepcopy
    y_unknown = y.loc[idxs_unknown_label]

    model = OPTICS(n_jobs=-1)
    model.fit(X_unknown)
    preds = model.fit_predict(X_unknown)

    # Allocate label
    for label in pd.value_counts(model.labels_).index:
        if label > -1:
            y_unknown.iloc[np.where(preds == label)[0]] = int(f"9999999{label}")

    # Process anomalies
    idxs_anomaly = np.where(model.labels_ == -1)[0]
    for idx, idx_anomaly in enumerate(idxs_anomaly, start=1):
        y_unknown.iloc[idx_anomaly] = int(f"-9999999{idx}")

    # Process return
    y_return = copy(y)
    y_return.loc[idxs_unknown_label] = y_unknown
    return y_return.astype('category')


def postprocess2_y(y):
    return pd.DataFrame(y, dtype=str).replace("^.*9999999.*$", "9999999", regex=True).astype('category')
