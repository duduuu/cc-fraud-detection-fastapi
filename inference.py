import polars as pl

from datetime import datetime, timedelta
from config import CFG

def predict_fraud(input_data: dict, dframes: dict, ml_models: dict, encoders: dict) -> float:
    df = pl.DataFrame([input_data])
    
    # 피처 엔지니어링 수행
    df = df.with_columns(
        (pl.col("User").cast(pl.Utf8) + "_" + pl.col("Card").cast(pl.Utf8)).alias("Card_ID")
    )
    
    df = df.with_columns([
        pl.col("Amount").str.replace(r"\$", "").cast(pl.Float64)
    ])
    
    df = df.with_columns(
        pl.when(pl.col("Amount") >= 1000)
        .then(1000)
        .otherwise(pl.col("Amount"))
        .alias("Amount")
    )

    df = df.with_columns([
        pl.col("Time").str.slice(0, 2).cast(pl.Int64).alias("Hour"),
        pl.col("Time").str.slice(3, 2).cast(pl.Int64).alias("Minute")
    ])

    df = df.with_columns([
        pl.datetime("Year", "Month", "Day", "Hour", "Minute")
        .alias("Datetime")
    ])
    df = df.with_columns([
        pl.col("Datetime").dt.weekday().alias("Weekday")
    ])

    df = df.with_columns([
        pl.col("Zip").cast(pl.Int64)
    ])
    
    unique_errors = [
        "Bad Zipcode",
        "Bad Card Number",
        "Bad Expiration",
        "Bad CVV",
        "Insufficient Balance",
        "Technical Glitch",
        "Bad PIN"
    ]

    df = df.with_columns(
        pl.when(
            pl.col("Errors").fill_null("").str.contains(error, literal=True)
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias(error.replace(" ", "_"))
        for error in unique_errors
    )
    
    df = df.with_columns(
        pl.when(pl.col("Merchant_State").is_null())
        .then(pl.lit("ONLINE"))
        .when(pl.col("Zip").is_null())
        .then(pl.lit("ABROAD"))
        .otherwise(pl.col("Merchant_State"))
        .alias("Merchant_State_New")
    )
    
    
    before_trans_df = dframes["trans"].filter(pl.col("Card_ID") == df["Card_ID"]).tail(1)
    if before_trans_df.is_empty():
        df = df.with_columns([
            pl.lit(-1).alias("Diff_Berfore_Trans_Time_Min"),
            pl.lit(0).alias("Before_City_Match"),
            pl.lit(0).alias("Before_State_Match"),
            pl.lit(0).alias("Before_Zip_Match")
        ])
    else:
        df = df.with_columns([
            (
                pl.col("Datetime") - before_trans_df["Datetime"]
            )
            .alias("Diff_Berfore_Trans_Time")
        ])
        df = df.with_columns([
            (pl.col("Diff_Berfore_Trans_Time").dt.total_seconds() / 60).fill_null(-1).cast(pl.Int32).alias("Diff_Berfore_Trans_Time_Min")
        ])
        
        df = df.with_columns([   
            pl.when((pl.col("Merchant_City") != before_trans_df["Merchant_City"]))
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("Before_City_Match"),
            
            pl.when((pl.col("Merchant_State") != before_trans_df["Merchant_State"]))
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("Before_State_Match"),
            
            pl.when((pl.col("Zip") != before_trans_df["Zip"]))
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("Before_Zip_Match")
        ])
    
    # 들어온 요청 업데이트
    dframes["trans"] = pl.concat([dframes["trans"], df.select(dframes["trans"].columns)])
    before_trans_df = dframes["trans"].filter(pl.col("Card_ID") == df["Card_ID"])
    
    period = {1:"24h", 7:"7d", 30:"30d"}
    for k, v in period.items():
        period_df = before_trans_df.filter(pl.col("Datetime") >= (df["Datetime"] - timedelta(days=k)))
        agg_df = period_df.select([
            pl.count().alias(f"period_count"),
            pl.col("Amount").mean().alias(f"amount_mean"),
            pl.col("Amount").max().alias(f"amount_max")
        ])
        df = df.with_columns([
            pl.lit(agg_df["period_count"][0]).alias(f"Count_{v}"),
            pl.lit(agg_df["amount_mean"][0]).alias(f"Amount_Mean_{v}"),
            pl.lit(agg_df["amount_max"][0]).alias(f"Amount_Max_{v}")
        ])
    
    df = (
        df
        .join(dframes["merchant"], on=["Merchant_Name"], how='left')
        .join(dframes["mcc"], on=["MCC"], how='left')
    )
        
    df = df.join(dframes["user"], on="User", how="left")
    df = df.join(dframes["card"], on="Card_ID", how="left")
    
    df = df.with_columns([
        (
            ((pl.col("Year") * 12) + pl.col("Month"))
            - ((pl.col("Birth_Year") * 12) + pl.col("Birth_Month"))
        )
        .floordiv(12)
        .alias("Age"),
        
        pl.when((pl.col("Merchant_City") != pl.col("User_City")))
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("User_City_Match"),
        
        pl.when((pl.col("Merchant_State_New") != pl.col("User_State")))
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("User_State_Match"),
        
        pl.when((pl.col("Zip") != pl.col("User_Zipcode")))
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("User_Zip_Match")
    ])
    
    df = df.with_columns([
        (
            ((pl.col("Year") * 12) + pl.col("Month"))
            - ((pl.col("Acct_Open_Year") * 12) + pl.col("Acct_Open_Month"))
        )
        .alias("Diff_Trans_Open"),
        
        (
            ((pl.col("Year") * 12) + pl.col("Month"))
            - ((pl.col("Expires_Year") * 12) + pl.col("Expires_Month"))
        )
        .alias("Diff_Trans_Expires"),
        
        (pl.col("Year") - pl.col("Year_PIN_last_Changed"))
        .alias("Diff_Year_PIN_last_Changed")
    ])
    
    df = df.with_columns([
        (pl.col("Amount") / (pl.col("Yearly_Income_Person") + 0.01))
            .alias("Amount_to_Income_Ratio"),
        (pl.col("Amount") / (pl.col("Per_Capita_Income_Zipcode") + 0.01))
            .alias("Amount_to_LocalIncome_Ratio")
    ])
    
    df = df.select(CFG.model_features[:-1])
    
    df[CFG.model_cat_features] = encoders["ordinal"].transform(df[CFG.model_cat_features]) 
     
    # 인퍼런스 수행
    preds = ml_models["lgbm"].predict(df)
    return preds[0]