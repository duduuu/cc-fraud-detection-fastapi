from datetime import datetime

class CFG:
    origin_trans_df_path = "data/credit_card_transactions-ibm_v2.csv"
    origin_user_df_path = "data/sd254_users.csv"
    origin_card_df_path = "data/sd254_cards.csv"
    
    train_path = "data/train_df_2015.parquet"
    test_path = "data/test_df_2015.parquet"
    model_path = "models/lightgbm.pkl"
    
    split_date = datetime(2019, 1, 1)
    
    output_log_path = "log/output.log"
    fastapi_log_path = "log/fastapi.log"
    
    online_trans_df_path = "data/trans_df.parquet"
    online_user_df_path = "data/user_df.parquet"
    online_card_df_path = "data/card_df.parquet"
    online_merchant_df_path = "data/merchant_df.parquet"
    online_mcc_df_path = "data/mcc_df.parquet"
    online_ordinal_encode_path = 'data/ordinal_encoder.pkl'
    online_model_path = "models/lightgbm.pkl"
    
    model_features = [
        "Month", "Weekday", "Hour", "Diff_Berfore_Trans_Time_Min", "Diff_Trans_Open", "Diff_Trans_Expires", "Count_24h", "Count_7d", "Count_30d", 
        "Amount", "Amount_Max_24h", "Amount_Mean_24h", "Amount_Max_7d", "Amount_Mean_7d", "Amount_Max_30d", "Amount_Mean_30d", "Amount_Max_Merchant", "Amount_Mean_Merchant",  "Amount_Max_MCC", "Amount_Mean_MCC",
        "Credit_Limit", "Per_Capita_Income_Zipcode", "Yearly_Income_Person", "Total_Debt", "Amount_to_Income_Ratio", "Amount_to_LocalIncome_Ratio",
        "Use_Chip", "Has_Chip", 
        "Before_City_Match", "User_City_Match", "Merchant_State_New", "Before_State_Match", "User_State_Match", "Before_Zip_Match", "User_Zip_Match",
        "MCC",
        'Age', 'Retirement_Age', 'Gender',
        "Num_Credit_Cards", "Card_Brand", "Card_Type", "Cards_Issued", "Diff_Year_PIN_last_Changed", "FICO_Score",
        "Bad_Expiration", "Bad_CVV", "Insufficient_Balance", "Technical_Glitch", "Bad_PIN", "Is_Fraud"
    ]
    model_cat_features = ["Gender", "Use_Chip", "Has_Chip", "Merchant_State_New", "Card_Brand", "Card_Type"]
    
    model = "lightgbm"
    
    lgbm_params = {
        'objective': 'binary',
        "metric": 'binary_logloss',
        'boosting_type': 'gbdt',
        'seed': 42,
        'verbose': -1
    }
    
    n_folds = 5
    num_boost_round = 3000
    seed = 42
    
    under_ratio = 0.005
    over_ratio = 0.5
    threshold = 0.5