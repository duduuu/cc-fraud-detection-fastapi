import polars as pl
import pandas as pd
import json
import logging

from sdv.metadata import Metadata, SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

from config import CFG

class DataSampler:
    def __init__(self):
        return
        
    def over_sampling(self, df: pd.DataFrame, over_ratio: float) -> pd.DataFrame:
        with open('metadata.json') as f:
            metadata_dict = json.load(f)
            
        df_fraud = df.loc[df["Is_Fraud"] == 1]
        
        n_syn = int(len(df_fraud) * over_ratio)
        
        df_fraud = df_fraud.drop(columns=["Is_Fraud"])
        
        metadata_features = {'columns': {key: metadata_dict['columns'][key] for key in CFG.model_features[:-1]}}
        metadata = SingleTableMetadata.load_from_dict(metadata_features)
        
        #synthesizer = CTGANSynthesizer.load(filepath=f"models/synthesizer.pkl")
        synthesizer = CTGANSynthesizer(metadata, epochs=500)
        synthesizer.fit(df_fraud)
        synthesizer.save(filepath="models/synthesizer.pkl")
        
        df_syn = synthesizer.sample(num_rows=n_syn)
        df_syn['Is_Fraud'] = 1
        
        df_syn = df_syn[CFG.model_features]
        
        df_sampled = pd.concat([df, df_syn], ignore_index=True) 
        
        return df_sampled
            
        
    def under_sampling(self, df: pd.DataFrame, under_ratio: float) -> pd.DataFrame:
        df_fraud = df.loc[df["Is_Fraud"] == 1]
        df_normal = df.loc[df["Is_Fraud"] == 0]

        n_fraud = len(df_fraud)
        n_normal = len(df_normal)

        n_normal_sample = int(n_fraud * (1 - under_ratio) / under_ratio)

        df_sampled = df_normal.sample(
            frac=n_normal_sample / n_normal,  
            random_state=CFG.seed
        )

        df_sampled = pd.concat([df_fraud, df_sampled], ignore_index=True) 
        
        return df_sampled