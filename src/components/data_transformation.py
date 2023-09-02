'''
This script aims to apply data cleaning and preprocessing to the data.
'''

# Debugging and verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# File handling.
import os

# Data manipulation.
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Modelling.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from category_encoders import TargetEncoder

# Utils.
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    '''
    Configuration class for data transformation.

    This data class holds configuration parameters related to data transformation. It includes attributes such as
    `preprocessor_file_path` that specifies the default path to save the preprocessor object file.

    Attributes:
        preprocessor_file_path (str): The default file path for saving the preprocessor object. By default, it is set
                                     to the 'artifacts' directory with the filename 'preprocessor.pkl'.

    Example:
        config = DataTransformationConfig()
        print(config.preprocessor_file_path)  # Output: 'artifacts/preprocessor.pkl'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    '''
    Data transformation class for preprocessing and transformation of train and test sets.

    This class handles the preprocessing and cleaning of datasets, including
    feature engineering and scaling.

    :ivar data_transformation_config: Configuration instance for data transformation.
    :type data_transformation_config: DataTransformationConfig
    '''
    def __init__(self) -> None:
        '''
        Initialize the DataTransformation instance with a DataTransformationConfig.
        '''
        self.data_transformation_config = DataTransformationConfig()


    def get_preprocessor(self):
        '''
        Get a preprocessor for data transformation.

        This method sets up pipelines for ordinal encoding, target encoding,
        and scaling of features.

        :return: Preprocessor object for data transformation.
        :rtype: ColumnTransformer
        :raises CustomException: If an exception occurs during the preprocessing setup.
        '''

        try:
            numerical_features = ['Age', 'Sex', 'Job', 'Credit amount', 'Duration']
            target_encoder_features = ['Saving accounts', 'Checking account']
            ordinal_encoder_features = ['Purpose', 'Housing']

            ordinal_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), 
                    ('ordinal_encoder', OrdinalEncoder()),
                    ('std_scaler', StandardScaler())
                    ]
                    )

            target_pipeline = Pipeline(
                steps=[
                    ('target_encoder',TargetEncoder(cols=target_encoder_features)), 
                    ('std_scaler', StandardScaler())
                    ]
                    )
            
            logging.info(f'Categorical features: {target_encoder_features+ordinal_encoder_features}.')
            logging.info(f'Target encoder will be applied to {target_encoder_features}.')
            logging.info(f'Ordinal encoder will be applied to: {ordinal_encoder_features}.')
            logging.info(f'Numerical features: {numerical_features}.')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('ordinal', ordinal_pipeline, ordinal_encoder_features),
                    ('target', target_pipeline, target_encoder_features),
                    ('std_scaler', StandardScaler(), numerical_features)
                    ], remainder='passthrough'
                    )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def apply_data_transformation(self, train_path, test_path):
        '''
        Apply data transformation process.

        Reads, preprocesses, and transforms training and test datasets.

        :param train_path: Path to the training dataset CSV file.
        :param test_path: Path to the test dataset CSV file.
        :return: Prepared training and test datasets and the preprocessor file path.
        :rtype: tuple
        :raises CustomException: If an exception occurs during the data transformation process.
        '''
        
        try:
            # obtaining train and test entire sets from artifacts.
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            logging.info('Read train and test sets.')
            logging.info('Obtaining preprocessor object.')

            preprocessor = self.get_preprocessor()

            # Expressing 'Sex' and 'Risk' (target) as binary features.
            train['Risk'] = train['Risk'].map({'bad': 1, 'good': 0})
            test['Risk'] = test['Risk'].map({'bad': 1, 'good': 0})
            train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
            test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})

            # Obtaining separate features for the steps below.
            target_feature = 'Risk'
            to_drop_feature = 'Unnamed: 0'

            # Getting train and test predictor and target sets.
            X_train = train.drop(columns=[target_feature, to_drop_feature])
            y_train = train[target_feature].copy()

            X_test = test.drop(columns=[target_feature, to_drop_feature])
            y_test = test[target_feature].copy()

            logging.info('Binarizing Sex and Target, dropping Target and irrelevant feature from train and test sets. X_train, X_test, y_train, y_test ready for preprocessing.')

            logging.info('Preprocessing train and test sets.')

            X_train_prepared = preprocessor.fit_transform(X_train, y_train)
            X_test_prepared = preprocessor.transform(X_test)

            # Getting final train and test entire prepared arrays.
            train_prepared = np.c_[
                X_train_prepared, np.array(y_train)
            ]
            test_prepared = np.c_[X_test_prepared, np.array(y_test)]

            logging.info('Entire train and test sets prepared.')

            logging.info('Save preprocessing object.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                object=preprocessor
            )
        
            return train_prepared, test_prepared, self.data_transformation_config.preprocessor_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
        


