import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "trained_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting data into train and test")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "RandomForest": RandomForestRegressor(),
                "LinearRegression": LinearRegression(), 
                "DecisionTree": DecisionTreeRegressor(),
                "KNeighbors": KNeighborsRegressor(),    
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.error(f"Best Model Score: {best_model_score} is less than 0.6")
                raise CustomException(f"Best Model Score: {best_model_score} is less than 0.6")

            logging.info(f"Best model on both train and test dataset is : {best_model_name}")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model)
            

            predicated = best_model.predict(X_test)
            r2_scor = r2_score(y_test, predicated)
            return r2_scor



        except Exception as e:
            logging.error(f"Error in Model Trainer: {str(e)}")
            raise CustomException(e,sys)