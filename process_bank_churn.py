import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, List, Any, Optional, Tuple


def select_input_columns(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> List[str]:
    """
    Select the list of input columns by dropping technical and irrelevant ones.

    Args:
        df (pd.DataFrame): The raw dataframe.
        target_col (str): Name of the target column.
        drop_cols (List[str]): Columns to exclude from the inputs.

    Returns:
        List[str]: List of input column names.
    """
    return [col for col in df.columns if col not in drop_cols + [target_col]]


def split_train_val(df: pd.DataFrame, target_col: str, test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the raw dataframe into training and validation sets with stratification.

    Args:
        df (pd.DataFrame): The raw dataframe.
        target_col (str): Name of the target column used for stratification.
        test_size (float): Share of the validation set.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state,
                            stratify=df[target_col])


def create_inputs_targets(train_df: pd.DataFrame, val_df: pd.DataFrame,
                          input_cols: List[str], target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for the training and validation sets.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        val_df (pd.DataFrame): Validation dataframe.
        input_cols (List[str]): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary with inputs and targets for train and val sets.
    """
    return {
        'train_inputs': train_df[input_cols].copy(),
        'train_targets': train_df[target_col].copy(),
        'val_inputs': val_df[input_cols].copy(),
        'val_targets': val_df[target_col].copy()
    }


def identify_column_types(inputs: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns of the input dataframe.

    Args:
        inputs (pd.DataFrame): Input dataframe.

    Returns:
        Tuple[List[str], List[str]]: Lists of numeric and categorical column names.
    """
    numeric_cols = inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = inputs.select_dtypes('object').columns.tolist()
    return numeric_cols, categorical_cols


def fit_imputer(inputs: pd.DataFrame, numeric_cols: List[str]) -> SimpleImputer:
    """
    Fit a mean imputer on the numeric columns of the training inputs.

    Args:
        inputs (pd.DataFrame): Training inputs.
        numeric_cols (List[str]): List of numeric columns.

    Returns:
        SimpleImputer: Fitted imputer.
    """
    return SimpleImputer(strategy='mean').fit(inputs[numeric_cols])


def fit_scaler(inputs: pd.DataFrame, numeric_cols: List[str]) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the numeric columns of the training inputs.

    Args:
        inputs (pd.DataFrame): Training inputs.
        numeric_cols (List[str]): List of numeric columns.

    Returns:
        MinMaxScaler: Fitted scaler.
    """
    return MinMaxScaler().fit(inputs[numeric_cols])


def fit_encoder(inputs: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Fit a one-hot encoder on the categorical columns of the training inputs.

    Args:
        inputs (pd.DataFrame): Training inputs.
        categorical_cols (List[str]): List of categorical columns.

    Returns:
        OneHotEncoder: Fitted encoder.
    """
    return OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(inputs[categorical_cols])


def impute_missing_values(inputs: pd.DataFrame, numeric_cols: List[str],
                          imputer: SimpleImputer) -> pd.DataFrame:
    """
    Impute missing numeric values with a fitted imputer.

    Args:
        inputs (pd.DataFrame): Inputs to transform.
        numeric_cols (List[str]): List of numeric columns.
        imputer (SimpleImputer): Fitted imputer.

    Returns:
        pd.DataFrame: Inputs with imputed numeric columns.
    """
    inputs[numeric_cols] = imputer.transform(inputs[numeric_cols])
    return inputs


def scale_numeric_features(inputs: pd.DataFrame, numeric_cols: List[str],
                           scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Scale numeric features with a fitted scaler.

    Args:
        inputs (pd.DataFrame): Inputs to transform.
        numeric_cols (List[str]): List of numeric columns.
        scaler (MinMaxScaler): Fitted scaler.

    Returns:
        pd.DataFrame: Inputs with scaled numeric columns.
    """
    inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
    return inputs


def encode_categorical_features(inputs: pd.DataFrame, categorical_cols: List[str],
                                encoder: OneHotEncoder) -> pd.DataFrame:
    """
    One-hot encode categorical features with a fitted encoder.

    Args:
        inputs (pd.DataFrame): Inputs to transform.
        categorical_cols (List[str]): List of categorical columns.
        encoder (OneHotEncoder): Fitted encoder.

    Returns:
        pd.DataFrame: Inputs with added encoded columns.
    """
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])
    return inputs


def preprocess_data(raw_df: pd.DataFrame, target_col: str = 'Exited',
                    drop_cols: Optional[List[str]] = None,
                    scaler_numeric: bool = True) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe: split, impute, scale and encode the features.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.
        target_col (str): Name of the target column.
        drop_cols (Optional[List[str]]): Columns to exclude from the inputs.
        scaler_numeric (bool): Whether to scale numeric features.

    Returns:
        Dict[str, Any]: Dictionary with X_train, train_targets, X_val, val_targets,
            input_cols, numeric_cols, categorical_cols, encoded_cols,
            imputer, scaler and encoder.
    """
    if drop_cols is None:
        drop_cols = ['id', 'CustomerId', 'Surname']

    input_cols = select_input_columns(raw_df, target_col, drop_cols)
    train_df, val_df = split_train_val(raw_df, target_col)
    data = create_inputs_targets(train_df, val_df, input_cols, target_col)

    numeric_cols, categorical_cols = identify_column_types(data['train_inputs'])

    imputer = fit_imputer(data['train_inputs'], numeric_cols)
    encoder = fit_encoder(data['train_inputs'], categorical_cols)
    scaler = fit_scaler(data['train_inputs'], numeric_cols) if scaler_numeric else None

    for split in ['train', 'val']:
        inputs = impute_missing_values(data[f'{split}_inputs'], numeric_cols, imputer)
        if scaler_numeric:
            inputs = scale_numeric_features(inputs, numeric_cols, scaler)
        data[f'{split}_inputs'] = encode_categorical_features(inputs, categorical_cols, encoder)

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    return {
        'X_train': data['train_inputs'][numeric_cols + encoded_cols],
        'train_targets': data['train_targets'],
        'X_val': data['val_inputs'][numeric_cols + encoded_cols],
        'val_targets': data['val_targets'],
        'input_cols': input_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'encoded_cols': encoded_cols,
        'imputer': imputer,
        'scaler': scaler,
        'encoder': encoder
    }


def preprocess_new_data(new_df: pd.DataFrame, numeric_cols: List[str],
                        categorical_cols: List[str], encoder: OneHotEncoder,
                        imputer: Optional[SimpleImputer] = None,
                        scaler: Optional[MinMaxScaler] = None) -> pd.DataFrame:
    """
    Preprocess new data with the transformers fitted on the training set.

    Args:
        new_df (pd.DataFrame): New raw data.
        numeric_cols (List[str]): List of numeric columns.
        categorical_cols (List[str]): List of categorical columns.
        encoder (OneHotEncoder): Encoder fitted on the training data.
        imputer (Optional[SimpleImputer]): Imputer fitted on the training data.
        scaler (Optional[MinMaxScaler]): Scaler fitted on the training data.

    Returns:
        pd.DataFrame: Processed inputs ready to be passed to the model.
    """
    inputs = new_df[numeric_cols + categorical_cols].copy()

    if imputer is not None:
        inputs = impute_missing_values(inputs, numeric_cols, imputer)
    if scaler is not None:
        inputs = scale_numeric_features(inputs, numeric_cols, scaler)

    inputs = encode_categorical_features(inputs, categorical_cols, encoder)
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    return inputs[numeric_cols + encoded_cols]
