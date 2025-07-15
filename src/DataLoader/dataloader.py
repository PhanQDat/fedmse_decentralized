import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path, header=None):
    dataframe = []
    for file in os.listdir(path):
        if ".csv" in file:
            filename = os.path.join(path, file)
            logging.info(f"Loading {filename}")
            dataframe.append(pd.read_csv(filename, header=header))
    dataframe = pd.concat(dataframe, ignore_index=True)
    return dataframe

class IoTDataProccessor(object):
    def __init__(self, scaler="standard"):
        if scaler == "standard":
            self.scaler = StandardScaler()
        
        if scaler == "minmax":
            self.scaler = MinMaxScaler((0, 1))

    def transform(self, dataframe, type="normal"):
        processed_data = self.scaler.transform(dataframe)
        if type == "normal":
            label = [0 for i in range(len(dataframe))]
        else:
            label = [1 for i in range(len(dataframe))]
        return processed_data, np.array(label)
    
    def fit_transform(self, dataframe):
        self.scaler = self.scaler.fit(dataframe)
        processed_data, label = self.transform(dataframe=dataframe, type="normal")
        return processed_data, label
        
    def get_metadata(self):
        metadata = {
            "mean": self.scaler.mean_,
            "std": self.scaler.scale_
        }
        return metadata
        
        
class IoTDataset(Dataset):
    """
    A custom Pytorch Dataset class for the N-BAIoT dataset.
    """
    
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx].astype(np.float32)
        y = self.label[idx].astype(np.float32)
        return X,y
    
    @property
    def input_dim_(self):
        return self.data.shape[1]
    
            
if __name__ == "__main__":
    device1_normal_path = "../../Data/N-BaIoT/Danmini_Doorbell/benign_traffic.csv"
    device1 = NBaIoTData(device1_normal_path)
    device1.preprocess_data()
    print(device1.dataframe)
    device1_dataset = NBaIoTDataset(device1.dataframe.values, device1.dataframe.values)
    
    print(device1_dataset.input_dim_)
    train_loader = DataLoader(
        dataset=device1_dataset,
        batch_size=100,
        num_workers=0,
        pin_memory=True
    )
    for batch in train_loader:
        print(batch[0].shape)
        print(batch[0])
        input()
    
