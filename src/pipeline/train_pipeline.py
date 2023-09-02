'''
This script aims to execute the entire training pipeline using the components. From data ingestion, transformation and model training.
'''

# Debugging and verbose.
import sys
from src.logger import logging
from src.exception import CustomException

# Components for data ingestion, transformation and model training.
from src.components import data_ingestion, data_transformation, model_trainer


class TrainPipeline:
    '''TrainPipeline class for training a machine learning pipeline.

    This class handles the training pipeline, including data ingestion,
    data transformation, and model training.

    Attributes:
        None

    Methods:
        __init__: Constructor method to initialize the class.
        train: Method to execute the training pipeline.

    '''

    def __init__(self) -> None:
        pass

    def train(self):
        '''Execute the training pipeline.

        This method performs the following steps:
        1. Data ingestion to obtain training and testing datasets.
        2. Data transformation for preprocessing.
        3. Model training using the best hyperparameters.

        It also prints the classification report and ROC-AUC score.

        Raises:
            CustomException: If an error occurs during the training pipeline.

        '''
        
        try:
        
            logging.info('Train full pipeline started.')

            logging.info('Splitting the data into train and test sets with data_ingestion component.')

            data_ingestion = DataIngestion()
            train, test = data_ingestion.apply_data_ingestion()

            logging.info('Train and test entire sets obtained (artifacts).')

            logging.info('Applying all the preprocessing steps required with data_transformation component.')
            
            data_transformation = DataTransformation()
            train_prepared, test_prepared, _ = data_transformation.apply_data_transformation(train, test)

            logging.info('Train and test entire prepared sets obtained.')

            logging.info('Train and save the best model using the best hyperparameters found during the modelling notebook analysis using model_trainer component.')
            
            model_trainer = ModelTrainer()

            logging.info('Final best model obtained (artifacts).')

            class_report, auc_score = model_trainer.apply_model_trainer(train_prepared, test_prepared)
            print('Final model classification report:')
            print(f'\n{class_report}')
            print(f'\nFinal model roc-auc score:')
            print(auc_score)

            logging.info('Classification report and ROC-AUC score presented.')

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    
    train_pipeline = TrainPipeline()
    train_pipeline.train()