from src.Mlproject.logger import logging
from src.Mlproject.exception import CustomException
import sys
from src.Mlproject.components.data_ingestion import DataIngestion
from src.Mlproject.components.data_ingestion import DataIngestionConfig
from src.Mlproject.components.data_transformation import DataTransformation
from src.Mlproject.components.model_trainer import ModelTrainer

if __name__=="__main__":
    logging.info('The execution has been started')
    try:
        # data_ingestion_conig=DataIngestionConfig
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()


        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
    