import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import logging
import joblib

from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb
import optuna

from sampling import DataSampler
from config import CFG

class FDS:
    def __init__(self, train_path: str, test_path: str):
             
        self.train_df = pd.read_parquet(train_path)
        self.test_df = pd.read_parquet(test_path)
        self.threshold = CFG.threshold
        self.seed = CFG.seed
    
    def train_model(
        self,
        model: str = "lightgbm",
        params: dict = CFG.lgbm_params,
        num_boost_round: int  = 1000,
        under_ratio: float = None,
        over_ratio: float = None,
        optuna_on: bool = False
        ):
        
        self.train_df, self.val_df = train_test_split(
            self.train_df,
            test_size=0.2,
            random_state=self.seed,
            stratify=self.train_df['Is_Fraud']
        )
        
        self.train_df = self.train_df[CFG.model_features]
        self.val_df = self.val_df[CFG.model_features]
        
        ds = DataSampler()
        if over_ratio:
            self.train_df = ds.over_sampling(self.train_df, over_ratio=over_ratio)
        if under_ratio:
            self.train_df = ds.under_sampling(self.train_df, under_ratio=under_ratio)
        
        print(f"ℹ️  Train Size : {len(self.train_df)}, {len(self.train_df.loc[self.train_df["Is_Fraud"] == 1])}, {len(self.train_df.loc[self.train_df["Is_Fraud"] == 1]) / len(self.train_df):.6f}")
        print(f"ℹ️  Val   Size : {len(self.val_df)}, {len(self.val_df.loc[self.val_df["Is_Fraud"] == 1])}, {len(self.val_df.loc[self.val_df["Is_Fraud"] == 1]) / len(self.val_df):.6f}")
        print(f"ℹ️  Test  Size : {len(self.test_df)}, {len(self.test_df.loc[self.test_df["Is_Fraud"] == 1])}, {len(self.test_df.loc[self.test_df["Is_Fraud"] == 1]) / len(self.test_df):.6f}")

        self.X_train = self.train_df.drop(columns=["Is_Fraud"])
        self.y_train = self.train_df["Is_Fraud"]
        
        self.X_val = self.val_df.drop(columns=["Is_Fraud"])
        self.y_val = self.val_df["Is_Fraud"]
        
        feature_importances = pd.DataFrame()
        feature_importances['feature'] = CFG.model_features[:-1]
        
        logging.info(f"========== LightGBM Test ==========")
        print(f"\n🚀  Training Started")

        training_start_time = time()
            
        self.train_data = lgb.Dataset(self.X_train, label=self.y_train)
        self.val_data = lgb.Dataset(self.X_val, label=self.y_val)
        
        # Optuna
        #best_params = {'learning_rate': 0.005, 'num_leaves': 31, 'max_depth': -1, 'min_data_in_leaf': 20}
        best_params = {'learning_rate': 0.08884294666024142, 'num_leaves': 51, 'max_depth': 7, 'min_data_in_leaf': 39, 'feature_fraction': 0.5555300484612417, 'bagging_fraction': 0.8709811138742957, 'bagging_freq': 5, 'lambda_l1': 0.45850208742319865, 'lambda_l2': 2.629620407771721}
        if optuna_on:
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial), n_trials=300)
            best_params = study.best_params
            print(best_params)
        
        params.update(best_params)
            
        model = lgb.train(params, self.train_data, num_boost_round=num_boost_round, valid_sets=[self.train_data, self.val_data])
        
        pred_proba = model.predict(self.X_val, num_iteration=model.best_iteration)
        pred_class = (pred_proba >= self.threshold).astype(int)
            
        pr = precision_score(self.y_val, pred_class)
        re = recall_score(self.y_val, pred_class)
        f1 = f1_score(self.y_val, pred_class)
        pr_auc = average_precision_score(self.y_val, pred_proba)

        joblib.dump(model, CFG.model_path)
            
        total_time = str(datetime.timedelta(seconds=time() - training_start_time))    
        
        print(f"🏁  Training Completed in {total_time}")
        print(f"📈  Validation Score: {model.best_score['valid_1']['binary_logloss']:.4f}, {pr:.4f}, {re:.4f}, {f1:.4f}, {pr_auc:.4f}")   
        logging.info(f"Validation Score: {model.best_score['valid_1']['binary_logloss']:.4f}, {pr:.4f}, {re:.4f}, {f1:.4f}, {pr_auc:.4f}")   
        
        feature_importances['average'] = model.feature_importance()
        feature_importances.to_csv(f'log/feature_importances_lgbm.csv')
        
        sns.set_theme()
        plt.figure(figsize=(24, 24))
        sns.barplot(data=feature_importances.sort_values(by='average', ascending=False), x='average', y='feature')
        plt.savefig(f'log/feature_importances_lgbm.png', dpi=300, bbox_inches='tight')
        
        
    def test_model(self):
        self.test_df = self.test_df[CFG.model_features]

        self.X_test = self.test_df.drop(columns=["Is_Fraud"])
        self.y_test = self.test_df['Is_Fraud']
        
        preds = np.zeros(len(self.X_test))

        print(f"\n🚀  Inference Started")
        inference_start_time = time()
    
        model = joblib.load(CFG.model_path)
            
        pred_proba = model.predict(self.X_test, num_iteration=model.best_iteration)
        pred_class = (pred_proba >= self.threshold).astype(int)
            
        pr = precision_score(self.y_test, pred_class)
        re = recall_score(self.y_test, pred_class)
        f1 = f1_score(self.y_test, pred_class)
        pr_auc = average_precision_score(self.y_test, pred_proba)

        total_time = str(datetime.timedelta(seconds=time() - inference_start_time))
        print(f"🏁  Inference Completed in {total_time}")
        print(f"📈  Test    Score:            {pr:.4f}, {re:.4f}, {f1:.4f}, {pr_auc:.4f}")    
        logging.info(f"Test    Score:         {pr:.4f}, {re:.4f}, {f1:.4f}, {pr_auc:.4f}")    
        
        self.test_df['pred_proba'] = pred_proba
        self.test_df.to_csv("log/result.csv", index=False)
        
        
    def objective(self, trial):
        trial_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 50),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
            'feature_pre_filter': False
        }
        params = CFG.lgbm_params
        params.update(trial_params)
        
        model = lgb.train(params, self.train_data, 1000, valid_sets=[self.train_data, self.val_data])
        
        pred_proba = model.predict(self.X_val, num_iteration=model.best_iteration)
        pred_class = (pred_proba >= self.threshold).astype(int)
              
        f1 = f1_score(self.y_val, pred_class)
        logging.info(f"F1 : {f1:.4f}, {params}")
        
        return f1
        
    
def main():
    logging.basicConfig(filename=CFG.output_log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    fd = FDS(
        train_path=CFG.train_path,
        test_path=CFG.test_path
    )
    
    fd.train_model(
        model=CFG.model,
        params=CFG.lgbm_params,
        num_boost_round=CFG.num_boost_round,
        under_ratio=CFG.under_ratio,
#        over_ratio =CFG.over_ratio,
        optuna_on=False
    )
    
    fd.test_model()
    
    
if __name__ == "__main__":
    main()