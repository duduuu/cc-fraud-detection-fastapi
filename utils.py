import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import torch

def fraud_ratio(df, col):
    count = df.group_by(col).agg(pl.count(col).alias("total_count"))
    fraud_count = df.filter(pl.col("Is_Fraud") == 1).group_by(col).agg(pl.count(col).alias("fraud_count"))

    count = count.join(fraud_count, on=col, how="left").sort(col)
    count = count.with_columns([(pl.col("fraud_count") / pl.col("total_count")).alias("fraud_ratio")])

    return count.to_pandas()

    plt.figure(figsize=(18, 4))
    sns.barplot(data=count_pd, x=col, y="fraud_ratio", color="red")
    plt.savefig(f'log/{filename}.png', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    
def plot_feature_distribution(df, col, filename, show=False):
    df_pd = df.select(col).to_pandas()
    
    plt.figure(figsize=(18, 4))
    sns.histplot(df_pd)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")
    plt.savefig(f'log/{filename}.png', dpi=300)
    
    if show:
        plt.show()
        
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)