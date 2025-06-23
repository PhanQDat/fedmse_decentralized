"""
This is training endpoint.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
@create date 2023-12-11 00:28:29
"""

from tqdm import tqdm
import torch
from torch import nn
import pickle
import os
import copy
import numpy as np
from sklearn.neighbors import KernelDensity
from DataLoader import IoTDataProccessor

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ClientTrainer(object):
    """
    Class for training a client model.

    Args:
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): Data loader for training data.
        loss_function (nn.Module, optional): The loss function to use. Defaults to nn.MSELoss.
        optimizer (torch.optim.Optimizer, optional): The optimizer to use. Defaults to torch.optim.Adam.
        epoch (int, optional): The number of epochs to train for. Defaults to 10.
        batch_size (int, optional): The batch size for training. Defaults to 100.
        lr_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 5.
        save_dir (str, optional): The directory to save the trained model and training tracking information. Defaults to "Checkpoint/ClientModel/".
        client_id (int, optional): Unique identifier for the client. Defaults to None.
        peers (list, optional): List of other ClientTrainer instances for peer-to-peer communication. Defaults to None.
        update_type (str, optional): Type of model aggregation ('avg', 'fusion_avg', 'mse_avg', 'fedprox'). Defaults to 'avg'.
        fedprox_mu (float, optional): Proximal term coefficient for FedProx. Defaults to 0.001.
    """

    def __init__(self, model=None, train_loader=None, loss_function=nn.MSELoss, optimizer=torch.optim.Adam,
                 epoch=10, batch_size=100, lr_rate=1e-3, update_type="avg",
                 patience=3, save_dir="Checkpoint/ClientModel/", fedprox_mu=0.001,
                 client_id=None, peers=None) -> None:
        
        if model is None:
            logging.info("Have to indicate the model to train.")
            return None
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logging.info("Created saving dir.")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.previous_global_model = copy.deepcopy(self.model)
        self.train_loader = train_loader
        self.loss_function = loss_function
        self.lr_rate = lr_rate
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr_rate)
        self.epoch = epoch
        self.batch_size = batch_size
        self.patience = patience
        self.save_dir = save_dir
        self.update_type = update_type
        self.fedprox_mu = fedprox_mu
        self.client_id = client_id
        self.peers = peers
        self.dev_dataset = None
        self.dev_kde_scores = None
    
    def save_model(self):
        """
        Save the trained model to the specified directory.
        """
        logging.info("Saving model to {}".format(self.save_dir))
        save_file = os.path.join(self.save_dir, "model.cpt")
        try:
            torch.save(
                self.model.state_dict(),
                save_file,
                _use_new_zipfile_serialization=False,
            )
        except:
            torch.save(self.model.state_dict(), save_file)
    
    def save_tracking_information(self):
        """
        Save the training tracking information.
        This method is currently empty and can be implemented as needed.
        """
        pass
    
    def create_dev_dataset(self, valid_loader):
        """
        Create a development dataset for fusion_avg aggregation.

        Args:
            valid_loader (torch.utils.data.DataLoader): Data loader for validation data.
        """
        logging.info(f"Client {self.client_id} creating development dataset...")
        self.dev_dataset = []
        for batch_input in valid_loader:
            self.dev_dataset.append(batch_input[0].cpu().numpy())
        self.dev_dataset = np.concatenate(self.dev_dataset, axis=0)
        
        data_processor = IoTDataProccessor(scaler="standard")
        self.dev_dataset, _ = data_processor.fit_transform(self.dev_dataset)
        
        if self.update_type == "fusion_avg":
            self.dev_kde_scores = KernelDensity(kernel='gaussian', bandwidth="scott") \
                .fit(self.dev_dataset).score_samples(self.dev_dataset)

    def run(self, train_loader, valid_loader=None):
        """
        Run the training process.

        Args:
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            valid_loader (torch.utils.data.DataLoader, optional): The data loader for validation data. Defaults to None.
        """
        if valid_loader is not None and self.update_type == "fusion_avg":
            self.create_dev_dataset(valid_loader)

        if self.update_type == "fedprox":
            print("Using FedProx")
            min_valid_loss = float("inf")
            worse_count = 0
            training_tracking = []
            for epoch in range(self.epoch):
                self.model.train()
                epoch_loss = 0
                for i, batch_input in zip(tqdm(range(len(train_loader)), desc='Training batch: ...'), train_loader):
                    _, _, loss = self.model(batch_input[0].to(self.device))
                    
                    # Add the proximal term to the loss
                    prox_term = 0.0
                    for param, global_param in zip(self.model.parameters(), self.previous_global_model.parameters()):
                        prox_term += torch.sum(torch.square(param - global_param.to(self.device)))
                    loss += self.fedprox_mu * prox_term
                    
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()
                    
                epoch_loss = epoch_loss / len(train_loader)
                if valid_loader is not None:
                    valid_loss = 0
                    self.model.eval()
                    with torch.no_grad():
                        for i, batch_input in zip(tqdm(range(len(valid_loader)), desc='Validating batch: ...'), valid_loader):
                            _, _, loss = self.model(batch_input[0].to(self.device))
                            
                            # Add the proximal term to the loss
                            prox_term = 0.0
                            for param, global_param in zip(self.model.parameters(), self.previous_global_model.parameters()):
                                prox_term += torch.sum(torch.square(param - global_param.to(self.device)))
                            loss += self.fedprox_mu * prox_term

                            valid_loss += loss.item()
                            
                        valid_loss = valid_loss / len(valid_loader)
                        training_tracking.append((epoch_loss, valid_loss))
                        logging.info(f"Epoch {epoch+1} - Training loss: {epoch_loss} - Validating loss: {valid_loss}")

                    if valid_loss < min_valid_loss:
                        min_valid_loss = valid_loss
                        self.save_model()
                        worse_count = 0
                    else:
                        worse_count += 1
                        if worse_count >= self.patience:
                            logging.info(f"Early stopping in epoch {epoch+1}.")
                            pickle.dump(training_tracking, open(os.path.join(self.save_dir, "training_tracking.pkl"), "wb"))
                            break
                
                pickle.dump(training_tracking, open(os.path.join(self.save_dir, "training_tracking.pkl"), "wb"))
        else:
            min_valid_loss = float("inf")
            worse_count = 0
            training_tracking = []
            for epoch in range(self.epoch):
                self.model.train()
                epoch_loss = 0
                for i, batch_input in zip(tqdm(range(len(train_loader)), desc='Training batch: ...'), train_loader):
                    _, _, loss = self.model(batch_input[0].to(self.device))
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()
                epoch_loss = epoch_loss / len(train_loader)
                if valid_loader is not None:
                    valid_loss = 0
                    self.model.eval()
                    with torch.no_grad():
                        for i, batch_input in zip(tqdm(range(len(valid_loader)), desc='Validating batch: ...'), valid_loader):
                            _, _, loss = self.model(batch_input[0].to(self.device))
                            valid_loss += loss.item()
                        
                        valid_loss = valid_loss / len(valid_loader)
                        training_tracking.append((epoch_loss, valid_loss))
                        logging.info(f"Epoch {epoch+1} - Training loss: {epoch_loss} - Validating loss: {valid_loss}")

                    if valid_loss < min_valid_loss:
                        min_valid_loss = valid_loss
                        self.save_model()
                        worse_count = 0
                    else:
                        worse_count += 1
                        if worse_count >= self.patience:
                            logging.info(f"Early stopping in epoch {epoch+1}.")
                            pickle.dump(training_tracking, open(os.path.join(self.save_dir, "training_tracking.pkl"), "wb"))
                            break
                
                pickle.dump(training_tracking, open(os.path.join(self.save_dir, "training_tracking.pkl"), "wb"))

    def share_model(self):
        """
        Share the current model state with other peers.

        Returns:
            tuple: (client_id, state_dict, num_samples)
        """
        logging.info(f"Client {self.client_id} sharing model...")
        return (self.client_id, self.model.state_dict(), self.batch_size * len(self.train_loader.dataset))

    def receive_models(self, selected_client_ids=None):
        """
        Receive models from peers, optionally limited to those in selected_client_ids.

        Args:
            selected_client_ids (list, optional): List of client IDs to receive models from. If None, receive from all peers.

        Returns:
            list: List of tuples (client_id, state_dict, num_samples) from peers.
        """
        logging.info(f"Client {self.client_id} receiving models from peers...")
        received_models = []
        peer_ids = []
        for peer in self.peers:
            if peer.client_id != self.client_id and (selected_client_ids is None or peer.client_id in selected_client_ids):
                peer_model = peer.share_model()
                received_models.append(peer_model)
                peer_ids.append(peer_model[0])  # peer_model[0] is client_id
        logging.info(f"Client {self.client_id} received models from peers: {peer_ids}")
        return received_models

    def fed_avg(self, models=None):
        """
        Perform federated averaging to aggregate the weights of local models.

        Args:
            models (list): List of tuples (client_id, state_dict, num_samples).

        Returns:
            None
        """
        logging.info(f"Client {self.client_id} performing FedAvg...")
        total_samples = sum(model[2] for model in models)
        avg_weights = {}
        for key in models[0][1].keys():
            avg_weights[key] = sum(model[1][key] * (model[2] / total_samples) for model in models)
        self.model.load_state_dict(avg_weights)

    def fusion_avg(self, models=None, valid_loader=None):
        """
        Perform fusion-based updating by calculating the average weights of local models.

        Args:
            models (list): List of tuples (client_id, state_dict, num_samples).
            valid_loader (torch.utils.data.DataLoader): Data loader for validation data.

        Returns:
            None
        """
        logging.info(f"Client {self.client_id} performing FusionAvg...")
        if self.dev_dataset is None:
            self.create_dev_dataset(valid_loader)
        update_weights = []
        weighted = []
        for i, (_, model_state, _) in enumerate(tqdm(models, desc='Calculating similarity...')):
            self.model.load_state_dict(model_state)
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                sim_score = -KernelDensity(kernel='gaussian', bandwidth="scott") \
                    .fit(generated_data.cpu().numpy()).score_samples(generated_data.cpu().numpy()).mean()
                weighted.append(1 / (sim_score + 1e-10))
                update_weights.append((model_state, 1 / (sim_score + 1e-10)))
        logging.info(f"Weighted scores: {weighted}")
        avg_weights = {}
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum(w[key] * alpha for w, alpha in update_weights) / sum(alpha for _, alpha in update_weights)
        self.model.load_state_dict(avg_weights)

    def fed_mse_avg(self, models=None, valid_loader=None):
        """
        Perform fusion-based updating by calculating the average weights of local models
        based on MSE loss of each AE-based model.

        Args:
            models (list): List of tuples (client_id, state_dict, num_samples).
            valid_loader (torch.utils.data.DataLoader): Data loader for validation data.

        Returns:
            None
        """
        logging.info(f"Client {self.client_id} performing FedMSEAvg...")
        if self.dev_dataset is None:
            self.create_dev_dataset(valid_loader)
        update_weights = []
        weighted = []
        for i, (_, model_state, _) in enumerate(tqdm(models, desc='Calculating similarity...')):
            self.model.load_state_dict(model_state)
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                sim_score = torch.nn.MSELoss(reduction='mean')(torch.Tensor(self.dev_dataset).to(self.device), generated_data)
                weighted.append(1 / (sim_score.item() + 1e-10))
                update_weights.append((model_state, 1 / (sim_score.item() + 1e-10)))
        logging.info(f"Weighted scores: {weighted}")
        avg_weights = {}
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum(w[key] * alpha for w, alpha in update_weights) / sum(alpha for _, alpha in update_weights)
        self.model.load_state_dict(avg_weights)

    def fedprox(self, models=None, mu=0.01):
        """
        Perform federated optimization using the FedProx algorithm to aggregate the weights of local models.

        Args:
            models (list): List of tuples (client_id, state_dict, num_samples).
            mu (float): Proximal term coefficient.

        Returns:
            None
        """
        logging.info(f"Client {self.client_id} performing FedProx...")
        total_samples = sum(model[2] for model in models)
        avg_weights = {}
        for key in models[0][1].keys():
            avg_weights[key] = sum(model[1][key] * (model[2] / total_samples) for model in models)
        self.model.load_state_dict(avg_weights)

    def aggregate_models(self, valid_loader, received_models):
        """
        Aggregate models from peers using the specified update_type.

        Args:
            valid_loader (torch.utils.data.DataLoader): Data loader for validation data.
            received_models (list): List of tuples (client_id, state_dict, num_samples) from peers.

        Returns:
            None
        """
        logging.info(f"Client {self.client_id} aggregating models with {self.update_type}...")
        own_model = self.share_model()
        all_models = [own_model] + received_models

        if self.update_type == "avg":
            self.fed_avg(all_models)
        elif self.update_type == "fusion_avg":
            self.fusion_avg(all_models, valid_loader)
        elif self.update_type == "mse_avg":
            self.fed_mse_avg(all_models, valid_loader)
        elif self.update_type == "fedprox":
            self.fedprox(all_models, mu=self.fedprox_mu)
        else:
            raise ValueError(f"Unsupported update_type: {self.update_type}")

        logging.info(f"Client {self.client_id} updated model with aggregated weights.")

    def decentralized_update(self, train_loader, valid_loader):
        """
        Perform a decentralized update by training locally, sharing models, and aggregating.

        Args:
            train_loader (torch.utils.data.DataLoader): Data loader for training data.
            valid_loader (torch.utils.data.DataLoader): Data loader for validation data.

        Returns:
            None
        """
        logging.info(f"Client {self.client_id} starting decentralized update...")
        self.run(train_loader, valid_loader)
        received_models = self.receive_models()
        self.aggregate_models(valid_loader, received_models)