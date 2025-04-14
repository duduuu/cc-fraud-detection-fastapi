import polars as pl

from datetime import datetime, timedelta

from config import CFG

from sklearn.preprocessing import OrdinalEncoder
import joblib

def trans_df_fe():
    trans_df = pl.read_csv(CFG.origin_trans_df_path)
    
    trans_df = trans_df.filter(pl.col("Year") >= 2015)
    
    
    trans_df = trans_df.with_columns([
        pl.col("Time").str.slice(0, 2).cast(pl.Int64).alias("Hour"),
        pl.col("Time").str.slice(3, 2).cast(pl.Int64).alias("Minute")
    ])
    
    trans_df = trans_df.with_columns([
        pl.datetime("Year", "Month", "Day", "Hour", "Minute")
        .alias("Datetime")
    ])
    
    # Online Request test
    trans_df.filter(pl.col("Datetime") >= CFG.split_date).drop("Datetime").write_parquet("data/test_df_request.parquet")
    
    trans_rename_dict = {
        'Use Chip': 'Use_Chip',
        'Merchant Name': 'Merchant_Name',
        'Merchant City': 'Merchant_City',
        'Merchant State': 'Merchant_State',
        'Errors?': 'Errors',
        'Is Fraud?': 'Is_Fraud'
    }
    trans_df = trans_df.rename(trans_rename_dict)

    trans_df = trans_df.with_columns(
        (pl.col("User").cast(pl.Utf8) + "_" + pl.col("Card").cast(pl.Utf8)).alias("Card_ID")
    )

    trans_df = trans_df.with_columns([
        pl.col("Amount").str.replace(r"\$", "").cast(pl.Float64),
        
        pl.col("Zip").cast(pl.Int64),
        
        pl.when(pl.col("Is_Fraud") == "Yes")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("Is_Fraud")
    ])
    
    # 임계치(1000) 이상 이상치 대체
    trans_df = trans_df.with_columns(
        pl.when(pl.col("Amount") >= 1000)
        .then(1000)
        .otherwise(pl.col("Amount"))
        .alias("Amount")
    )
    
    trans_df = trans_df.with_columns([
        pl.col("Datetime").dt.weekday().alias("Weekday")
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
    
    # 에러 종류별 피처 생성
    trans_df = trans_df.with_columns(
        pl.when(
            pl.col("Errors").fill_null("").str.contains(error, literal=True)
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias(error.replace(" ", "_"))
        for error in unique_errors
    )

    trans_df = trans_df.with_columns(
        (
            pl.col("Bad_Zipcode") +
            pl.col("Bad_Card_Number") +
            pl.col("Bad_Expiration") +
            pl.col("Bad_CVV") +
            pl.col("Insufficient_Balance") +
            pl.col("Technical_Glitch") +
            pl.col("Bad_PIN")
        ).alias("Error_Count")
    )
    
    # Merchant State 결측치 처리한 새로운 피처 생성
    trans_df = trans_df.with_columns(
        pl.when(pl.col("Merchant_State").is_null())
        .then(pl.lit("ONLINE"))
        .when(pl.col("Zip").is_null())
        .then(pl.lit("ABROAD"))
        .otherwise(pl.col("Merchant_State"))
        .alias("Merchant_State_New")
    )
    
    # 직전 거래와의 시간 간격, 위치 일치 여부
    trans_df = trans_df.with_columns([
        (pl.col("Datetime") - pl.col("Datetime").shift(1)).over("Card_ID").alias("Diff_Berfore_Trans_Time")
    ])
    trans_df = trans_df.with_columns([
        (pl.col("Diff_Berfore_Trans_Time").dt.total_seconds() / 60).fill_null(-1).cast(pl.Int32).alias("Diff_Berfore_Trans_Time_Min")
    ])

    trans_df = trans_df.with_columns([   
        pl.when((pl.col("Merchant_City") != pl.col("Merchant_City").shift(1)).over("Card_ID"))
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("Before_City_Match"),
        
        pl.when((pl.col("Merchant_State") != pl.col("Merchant_State").shift(1)).over("Card_ID"))
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("Before_State_Match"),
        
        pl.when((pl.col("Zip") != pl.col("Zip").shift(1)).over("Card_ID"))
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("Before_Zip_Match")
    ])

    # 통계 피처 생성
    def grou_by_rolling(df, period):
        df_ = (
            df.rolling(
                "Datetime",
                group_by="Card_ID",
                period=period,
            )
            .agg([
                pl.len().alias(f"Count_{period}"),
                pl.col("Amount").max().alias(f"Amount_Max_{period}"),
                pl.col("Amount").mean().alias(f"Amount_Mean_{period}")
            ])
        )
        return df_

    df_24h = grou_by_rolling(trans_df, "24h")
    df_7d = grou_by_rolling(trans_df, "7d")
    df_30d = grou_by_rolling(trans_df, "30d")
    
    df_agg_mer = (
        trans_df.filter(pl.col("Datetime") < CFG.split_date) # Test 데이터셋 제외
        .group_by("Merchant_Name")
        .agg([
            pl.col("Amount").max().alias("Amount_Max_Merchant"),
            pl.col("Amount").mean().alias("Amount_Mean_Merchant")
        ])
    )
    df_agg_mer.write_parquet("data/merchant_df.parquet")
    df_agg_mcc = (
        trans_df.filter(pl.col("Year") < CFG.split_date)
        .group_by("MCC")
        .agg([
            pl.col("Amount").max().alias("Amount_Max_MCC"),
            pl.col("Amount").mean().alias("Amount_Mean_MCC")
        ])
    )
    df_agg_mcc.write_parquet("data/mcc_df.parquet")

    trans_df = (
        trans_df
        .join(df_24h, on=["Card_ID", "Datetime"], how="left")
        .join(df_7d, on=["Card_ID", "Datetime"], how="left")
        .join(df_30d, on=["Card_ID", "Datetime"], how="left")
        .join(df_agg_mer, on=["Merchant_Name"], how='left')
        .join(df_agg_mcc, on=["MCC"], how='left')
    )

    features = [
        "User", "Card_ID", "Datetime", "Year", "Month", "Weekday", "Day", "Hour", "Diff_Berfore_Trans_Time_Min", "Count_24h", "Count_7d", "Count_30d", 
        "Amount", "Amount_Max_24h", "Amount_Mean_24h", "Amount_Max_7d", "Amount_Mean_7d", "Amount_Max_30d", "Amount_Mean_30d", "Amount_Max_Merchant", "Amount_Mean_Merchant", "Amount_Max_MCC", "Amount_Mean_MCC",
        "Use_Chip", "Merchant_Name", "Merchant_City", "Before_City_Match", "Merchant_State", "Before_State_Match", "Merchant_State_New", "Zip", "Before_Zip_Match", "MCC", "Error_Count"
        ] + [error.replace(" ", "_") for error in unique_errors] + ["Is_Fraud"]
    
    trans_df = trans_df.select(features)
    trans_df.write_parquet("data/trans_df_2015.parquet")
    
    # Online Trans
    trans_df2 = trans_df.select(["User", "Card_ID", "Datetime", "Amount", "Merchant_Name", "Merchant_City", "Merchant_State", "Zip"])
    trans_df2 = trans_df2.filter(pl.col("Datetime") < CFG.split_date)

    df1 = trans_df2.filter(pl.col("Datetime") >= (CFG.split_date - timedelta(days=30)))
    df2 = trans_df2.group_by(["User", "Card_ID"]).tail(1)

    df_concat = pl.concat([df1, df2]).unique().sort(["Datetime"])

    df_concat.write_parquet("data/trans_df.parquet")
    
    return trans_df

def user_df_fe():
    user_df = pl.read_csv(CFG.origin_user_df_path)
    
    user_rename_dict = {
        'Current Age': 'Current_Age', 
        'Retirement Age': 'Retirement_Age', 
        'Birth Year': 'Birth_Year', 
        'Birth Month': 'Birth_Month', 
        'City': 'User_City', 
        'State': 'User_State', 
        'Zipcode': 'User_Zipcode', 
        'Per Capita Income - Zipcode': 'Per_Capita_Income_Zipcode', 
        'Yearly Income - Person': 'Yearly_Income_Person', 
        'Total Debt': 'Total_Debt', 
        'FICO Score': 'FICO_Score', 
        'Num Credit Cards': 'Num_Credit_Cards'
    }

    user_df = user_df.rename(user_rename_dict)
    
    user_df = user_df.with_row_index("User")

    amount_cols = ["Per_Capita_Income_Zipcode", "Yearly_Income_Person", "Total_Debt"]
    user_df = user_df.with_columns([
        pl.col(col).str.replace(r"\$", "").cast(pl.Int64)
        for col in amount_cols
    ])
        
    user_df = user_df.with_columns(
        pl.col("Apartment").fill_null(0).alias("Apartment")
    )
    
    user_df = user_df.with_columns(
        pl.when(pl.col("FICO_Score") < 580)
        .then(0)
        .when(pl.col("FICO_Score") < 670)
        .then(1)
        .when(pl.col("FICO_Score") < 740)
        .then(2)
        .when(pl.col("FICO_Score") < 800)
        .then(3)
        .otherwise(4)
        .alias("FICO_Score_Rank")
    )
    
    features = ['User', 'Retirement_Age', 'Birth_Year', 'Birth_Month', 'Gender', 'Apartment', 'User_City', 'User_State', 'User_Zipcode', 
                'Per_Capita_Income_Zipcode', 'Yearly_Income_Person', 'Total_Debt', 
                'FICO_Score', 'FICO_Score_Rank', 'Num_Credit_Cards']

    user_df = user_df[features]
    user_df.write_parquet("data/user_df.parquet")
    
    return user_df
    
def card_df_fe():
    card_df = pl.read_csv(CFG.origin_card_df_path)
    
    card_rename_dict = {
        "CARD INDEX": "Card",
        "Card Brand": "Card_Brand",
        "Card Type": "Card_Type",
        "Card Number": "Card_Number",
        "Has Chip": "Has_Chip",
        "Cards Issued": "Cards_Issued",
        "Credit Limit": "Credit_Limit",
        "Acct Open Date": "Acct_Open_Date",
        "Year PIN last Changed": "Year_PIN_last_Changed",
        "Card on Dark Web": "Card_on_Dark_Web",
    }

    card_df = card_df.rename(card_rename_dict)
    
    card_df = card_df.with_columns(
        (pl.col("User").cast(pl.Utf8) + "_" + pl.col("Card").cast(pl.Utf8)).alias("Card_ID")
    )

    card_df = card_df.with_columns([
        pl.col("Credit_Limit").str.replace(r"\$", "").cast(pl.Int64)
    ])

    card_df = card_df.with_columns([
        pl.col("Expires").str.slice(0, 2).cast(pl.Int64).alias("Expires_Month"),
        pl.col("Expires").str.slice(3, 4).cast(pl.Int64).alias("Expires_Year"),
        pl.col("Acct_Open_Date").str.slice(0, 2).cast(pl.Int64).alias("Acct_Open_Month"),
        pl.col("Acct_Open_Date").str.slice(3, 4).cast(pl.Int64).alias("Acct_Open_Year"),
    ])
    
    features =['Card_ID', 'Card_Brand', 'Card_Type', 'Has_Chip', 'Cards_Issued', 'Credit_Limit', 'Year_PIN_last_Changed', 
               'Expires_Year', 'Expires_Month', 'Acct_Open_Year', 'Acct_Open_Month']
    
    card_df = card_df[features]
    card_df.write_parquet("data/card_df.parquet")
    
    return card_df

def total_df_fe(trans_df: pl.DataFrame, user_df: pl.DataFrame, card_df: pl.DataFrame):
    
    df = trans_df.join(user_df, on="User", how="left")
    df = df.join(card_df, on="Card_ID", how="left")

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
        pl.when((pl.col("Age") >= pl.col("Retirement_Age")))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("Retired")
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
            .alias("Amount_to_LocalIncome_Ratio"),
        (pl.col("Total_Debt") / (pl.col("Yearly_Income_Person") + 0.01))
            .alias("Debt_to_Income"),
        (pl.col("Total_Debt") / (pl.col("Credit_Limit") + 0.01))
            .alias("Debt_to_CreditLimit"),
        ((pl.col("Amount") + pl.col("Total_Debt")) / (pl.col("Yearly_Income_Person") + 0.01))
            .alias("TotalBurden_to_Income"),
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
        
    features = [
        "User", "Card_ID", 
        "Datetime", "Year", "Month", "Weekday", "Hour", "Diff_Berfore_Trans_Time_Min", "Diff_Trans_Open", "Diff_Trans_Expires", "Count_24h", "Count_7d", "Count_30d", 
        "Amount", "Amount_Max_24h", "Amount_Mean_24h", "Amount_Max_7d", "Amount_Mean_7d", "Amount_Max_30d", "Amount_Mean_30d", "Amount_Max_Merchant", "Amount_Mean_Merchant", "Amount_Max_MCC", "Amount_Mean_MCC",
        "Credit_Limit", "Per_Capita_Income_Zipcode", "Yearly_Income_Person", "Total_Debt", "Amount_to_Income_Ratio", "Amount_to_LocalIncome_Ratio", "Debt_to_Income", "Debt_to_CreditLimit", "TotalBurden_to_Income",
        "Use_Chip", "Has_Chip", 
        "Merchant_Name", "Merchant_City", "Before_City_Match", "User_City_Match", "Merchant_State", "Before_State_Match", "User_State_Match", "Merchant_State_New", "Zip", "Before_Zip_Match", "User_Zip_Match",
        "MCC",
        'Age', 'Retirement_Age', 'Retired', 'Gender', "Apartment",
        "Num_Credit_Cards", "Card_Brand", "Card_Type", "Cards_Issued", "Diff_Year_PIN_last_Changed", "FICO_Score", "FICO_Score_Rank", "Error_Count"
        ] + [error.replace(" ", "_") for error in unique_errors] + ["Is_Fraud"]

    df = df.select(features)

    categorical_cols = ["Gender", "Use_Chip", "Has_Chip", "Merchant_Name", "Merchant_City", "Merchant_State", "Merchant_State_New", "Zip", "Card_Brand", "Card_Type"]

    df = df.with_columns([
        pl.col(col).cast(pl.String).cast(pl.Categorical)
        for col in categorical_cols
    ])

    #df.write_parquet("data/df_all.parquet")
    return df

def train_test_split(df: pl.DataFrame):
    train_df = df.filter(pl.col("Datetime") < CFG.split_date)
    test_df = df.filter(pl.col("Datetime") >= CFG.split_date)
    
    train_df = train_df.to_pandas()
    test_df = test_df.to_pandas()
    
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_df[CFG.model_cat_features] = oe.fit_transform(train_df[CFG.model_cat_features])
    feature_order = train_df.columns.tolist()
    
    test_df[CFG.model_cat_features] = oe.transform(test_df[CFG.model_cat_features])
    test_df = test_df[feature_order]
    
    joblib.dump(oe, CFG.online_ordinal_encode_path)
    
    train_df.to_parquet(CFG.train_path, index=False)
    test_df.to_parquet(CFG.test_path, index=False)
    
    
def main():
    trans_df = trans_df_fe()
    #trans_df = pl.read_parquet("data/trans_df.parquet")
    print(f'✅  Transaction FE Done..')
    
    user_df = user_df_fe()
    #user_df = pl.read_parquet("data/user_df.parquet")
    print(f'✅  User FE Done..')
    
    card_df = card_df_fe()
    #card_df = pl.read_parquet("data/card_df.parquet")
    print(f'✅  Card FE Done..')
    
    df = total_df_fe(trans_df, user_df, card_df)
    print(f'✅  Total FE Done..')
    
    train_test_split(df)
    print(f'✅  Train / Test Split Done..')
    
    
if __name__ == "__main__":
    main()