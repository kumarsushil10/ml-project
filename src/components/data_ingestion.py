import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", 'train.csv')
    test_data_path: str = os.path.join("artifacts", 'test.csv')
    raw_data_path: str = os.path.join("artifacts", 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingection_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion Started")
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Data Ingestion Completed")

            os.makedirs(os.path.dirname(self.ingection_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingection_config.raw_data_path, index=False,header=True)
            logging.info(f"Raw Data Saved at {self.ingection_config.raw_data_path}")
            
            logging.info("Train Test Split Started")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train Test Split Completed")

            train_set.to_csv(self.ingection_config.train_data_path, index=False,header=True)
            logging.info(f"Train Data Saved at {self.ingection_config.train_data_path}")

            test_set.to_csv(self.ingection_config.test_data_path, index=False,header=True)
            logging.info(f"Test Data Saved at {self.ingection_config.test_data_path}")

            return(
                self.ingection_config.train_data_path,
                self.ingection_config.test_data_path
                )
        except Exception as e:
            logging.error(f"Data Ingestion Failed: {str(e)}")
            raise CustomException(e,sys)
        


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()