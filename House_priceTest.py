import torch 
import torch.nn as nn
import sklearn.preprocessing as process
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import pandas as pd 
import copy
import tqdm
import matplotlib.pyplot as plt
import wandb

df_test = pd.read_csv('House_price/Test.csv')
df_train = pd.read_csv('House_price/Train.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = process.MinMaxScaler(feature_range=(0,100))
def proceess_dataframe(df:pd.DataFrame , Train:bool) -> pd.DataFrame: 
    """
    This function is used to process the dataframe by make a
    numerical representation of a string and scaling
    all of the important values
    """
    if not isinstance(df,pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame")
    df  = copy.deepcopy(df)
    for idx,rows in df.iterrows():
        if rows["POSTED_BY"] == "Owner":
            df.at[idx,"POSTED_BY"] = 0
        elif rows["POSTED_BY"] == "Dealer": 
            df.at[idx,"POSTED_BY"] = 1
        else: 
            df.at[idx,"POSTED_BY"] = 2
            
        if rows["BHK_OR_RK"] == "BHK":
            df.at[idx,"BHK_OR_RK"] = 0
        else: 
            df.at[idx,"BHK_OR_RK"] = 1
    df.drop(columns=["ADDRESS","LATITUDE","LONGITUDE"],inplace=True)
    if Train: 
        df.rename(columns={"TARGET(PRICE_IN_LACS)":"PRICE"},inplace=True)
        df[["PRICE","SQUARE_FT"]] = scaler.fit_transform(df[["PRICE","SQUARE_FT"]]) 
    else: 
        df[["SQUARE_FT"]] = scaler.fit_transform(df[["SQUARE_FT"]])
    return df

shifted_df_Test = proceess_dataframe(df_test,Train=False).to_numpy()
x_test = torch.from_numpy(shifted_df_Test.astype(float)).to(device=device,dtype=torch.float32)

class TestDataset(Dataset):
    def __init__(self,x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        return self.x[idx]

test_dataset = TestDataset(x_test)
test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)

class Regression(nn.Module): 
    def __init__(self,hidden_layer):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=8,out_features=hidden_layer),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer,out_features=hidden_layer),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer,out_features=hidden_layer),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer,out_features=1),
        )
    def forward(self,x):
        return self.layer(x)
    
model = Regression(hidden_layer=20).to(device=device)
model.load_state_dict(torch.load("modelHouse.pth"))

predicted_price = []
for x in test_loader:
    model.eval()
    with torch.inference_mode():
        prediction = model(x).cpu().numpy()
        scaler.fit(df_train[["TARGET(PRICE_IN_LACS)"]])
        prediction = scaler.inverse_transform(prediction)
        predicted_price.append(prediction)
df_test["TARGET(PRICE_IN_LACS)"] = predicted_price
df_test.to_csv("predicted_price.csv")
print(predicted_price)