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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_train = pd.read_csv('House_price/Train.csv')
df_test = pd.read_csv('House_price/Test.csv')

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
    scaler = process.MinMaxScaler(feature_range=(0,100))
    if Train: 
        df.rename(columns={"TARGET(PRICE_IN_LACS)":"PRICE"},inplace=True)
        df[["PRICE","SQUARE_FT"]] = scaler.fit_transform(df[["PRICE","SQUARE_FT"]]) 
    else: 
        df[["SQUARE_FT"]] = scaler.fit_transform(df[["SQUARE_FT"]])
    return df

shifted_df_Train = proceess_dataframe(df_train,Train=True).to_numpy()
shifted_df_Test = proceess_dataframe(df_test,Train=False).to_numpy()

x_train = torch.from_numpy(shifted_df_Train[:,:-1].astype(float)).to(device=device,dtype=torch.float32)
y_train = torch.from_numpy(shifted_df_Train[:,-1].astype(float)).to(device=device,dtype=torch.float32).unsqueeze(1)
x_test = torch.from_numpy(shifted_df_Test[:,:-1].astype(float)).to(device=device,dtype=torch.float32)
y_test = torch.from_numpy(shifted_df_Test[:,-1].astype(float)).to(device=device,dtype=torch.float32).unsqueeze(1)

class HousePrice(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y 
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = HousePrice(x_train,y_train)
test_dataset = HousePrice(x_test,y_test)
train_loader = DataLoader(train_dataset,batch_size=100,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=100,shuffle=False)

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
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.MSELoss()
stepLR = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)
NUM_EPOCHS = 200

wandb.init(project="HousePrice",config={"epochs":NUM_EPOCHS,"batch_size":100,"learning_rate":0.001,"LR step":"0.1/100"})
wandb.watch(model,log="all")
for i in tqdm.tqdm(range(NUM_EPOCHS)):
    try: 
        model.train() 
        for x,y in train_loader:
            prediction = model(x)
            train_loss = loss_fn(prediction,y)
            optimizer.zero_grad()
            train_loss.backward()
            wandb.log({"Loss":train_loss.item()})
            optimizer.step()
        stepLR.step()
    except KeyboardInterrupt: 
        torch.save(model.state_dict(),"model.pth")
        break
torch.save(model.state_dict(),"modelHouse.pth")