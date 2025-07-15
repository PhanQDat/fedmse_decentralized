import os
import json
import pickle   
import pandas as pd
import numpy as np
import torch
import argparse
import copy
import random
from torch.utils.data import DataLoader, random_split, ConcatDataset
from Model import Shrink_Autoencoder
from Model import Autoencoder
from DataLoader import load_data
from DataLoader import IoTDataset
from DataLoader import IoTDataProccessor
from Trainer import ClientTrainer
from Evaluator import Evaluator

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')

num_participants = 0.5
epoch = 20
num_rounds = 3
lr_rate = 1e-5
shrink_lambda = 10
network_size = 10
data_seed = 1234
no_Exp = f"IID-Update_Exp6_scale_{epoch}epoch_{network_size}client_{num_rounds}rounds_lr{lr_rate}_lamda{shrink_lambda}_ratio{num_participants*100}_dataseed{data_seed}"

num_runs = 2
batch_size = 12

new_device = True
min_val_loss = float("inf")
global_patience = 1
global_worse = 0
metric = "AUC" #AUC or classification
dim_features = 115   #nba-iot: 115; cic-2023: 46
    
scen_name = 'FL-IoT' 

config_file = f"Configuration/kitsune-iot-10clients.json"
# config_file = f"Configuration/kitsune-iot-10clients-IID.json"
# config_file = f"Configuration/kitsune-iot-10clients-nonIID.json"

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_mse_score(model, valid_loader, device, client_id):
    """
    Compute MSE score for a model on validation data.

    Args:
        model (nn.Module): Model to evaluate.
        valid_loader (DataLoader): Validation data loader.
        device (torch.device): Device to run computation on.
        client_id (int): Client ID for logging.

    Returns:
        float: Average MSE score.
    """
    model.eval()
    mse_loss = 0.0
    with torch.no_grad():
        for batch_input in valid_loader:
            if torch.isnan(batch_input[0]).any() or torch.isinf(batch_input[0]).any():
                logging.warning(f"Invalid data detected in valid_loader for client {client_id}")
                return float("inf")
            _, generated_data, _ = model(batch_input[0].to(device))
            if torch.isnan(generated_data).any() or torch.isinf(generated_data).any():
                logging.warning(f"Invalid generated data detected for client {client_id}")
                return float("inf")
            mse = torch.nn.MSELoss(reduction='mean')(batch_input[0].to(device), generated_data)
            mse_loss += mse.item()
    mse_loss /= len(valid_loader)
    return mse_loss

if __name__ == "__main__":
    random.seed(data_seed)
    np.random.seed(data_seed)
    try:
        logging.info("Loading configuration...")
        with open(config_file, "r") as config_file:
            config = json.load(config_file)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        exit(1)
        
    devices_list = random.sample(config['devices_list'], network_size)
    client_info = []
    for device in devices_list:
        logging.info(f"Creating metadata for client {device['name']}...")
        normal_data_path = os.path.join(config['data_path'], device["normal_data_path"])
        abnormal_data_path = os.path.join(config['data_path'], device["abnormal_data_path"])
        test_new_normal_data_path = os.path.join(config['data_path'], device["test_normal_data_path"])
        
        # Check if data files exist
        for path in [normal_data_path, abnormal_data_path, test_new_normal_data_path]:
            if not os.path.exists(path):
                logging.error(f"Data file not found: {path}")
                exit(1)
        
        logging.info(f"Loading data from {device['name']}...")
        
        normal_data = load_data(normal_data_path)
        normal_data = normal_data.sample(frac=1).reset_index(drop=True)
        abnormal_data = load_data(abnormal_data_path)
        abnormal_data = abnormal_data.sample(frac=1).reset_index(drop=True)
        
        if new_device:
            new_normal_data = load_data(test_new_normal_data_path)
        
        device_name = device['name']
        print(f"{device_name} has {len(normal_data)} normal data and {len(abnormal_data)} abnormal data")
        
        train_normal_size = int(0.4 * len(normal_data))
        valid_normal_size = int(0.1 * len(normal_data))
        dev_normal_size = int(0.4 * len(normal_data))
        test_normal_size = len(normal_data) - train_normal_size - valid_normal_size - dev_normal_size
        
        train_normal_data = normal_data[:train_normal_size]
        valid_normal_data = normal_data[train_normal_size:train_normal_size+valid_normal_size]
        dev_normal_data = normal_data[train_normal_size+valid_normal_size:train_normal_size+valid_normal_size+dev_normal_size]
        test_normal_data = normal_data[train_normal_size+valid_normal_size+dev_normal_size:]

        # Create a separate IoTDataProccessor for each client
        data_processor = IoTDataProccessor(scaler="standard")
        processed_train_data, train_label = data_processor.fit_transform(train_normal_data)
        processed_valid_data, valid_label = data_processor.transform(valid_normal_data)
        processed_test_data, test_label = data_processor.transform(test_normal_data)
        processed_abnormal_data, abnormal_label = data_processor.transform(abnormal_data, type="abnormal")
        
        if new_device:
            processed_new_normal_data, new_normal_label = data_processor.transform(new_normal_data)
            processed_test_data = np.concatenate([processed_test_data, processed_new_normal_data], axis=0)
            processed_test_label = np.concatenate([test_label, new_normal_label], axis=0)
            test_dataset = IoTDataset(processed_test_data, processed_test_label)
        else:
            test_dataset = IoTDataset(processed_test_data, test_label)
        
        train_dataset = IoTDataset(processed_train_data, train_label)
        valid_dataset = IoTDataset(processed_valid_data, valid_label)
        abnormal_dataset = IoTDataset(processed_abnormal_data, abnormal_label)
        
        test_dataset = ConcatDataset([test_dataset, abnormal_dataset])

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            pin_memory=True
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            pin_memory=True
        )
        
        client_info.append({
            "device": device['name'],
            "save_dir": "",
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "test_loader": test_loader,
            "test_dataset": (processed_test_data, test_label),
            "dev_normal_dataset": dev_normal_data
        })

    for update_type in ["avg", "fedprox", "mse_avg"]:
        for model_type in ["hybrid", "autoencoder"]:
            for run in range(num_runs):
                logging.info(f"Starting model_type: {model_type}, update_type: {update_type}, run: {run}")                
                set_seeds(run*10000)
                for client in client_info:
                    client['save_dir'] = os.path.join(f"Checkpoint/{network_size}/{no_Exp}/{run}/ClientModel", scen_name, model_type, update_type, client['device'])
                
                global_worse = 0
                min_val_loss = float("inf")
                
                directory = f'Checkpoint/Results/Update/{network_size}/{no_Exp}/Run_{run}/{metric}'
                os.makedirs(directory, exist_ok=True)

                filename = f'{directory}/{scen_name}_{num_participants}_{model_type}_{update_type}_results.json'
                open(filename, 'w').close()
                
                if model_type == "hybrid":
                    global_model = Shrink_Autoencoder(input_dim=dim_features,
                                                    output_dim=dim_features,
                                                    shrink_lambda=shrink_lambda,
                                                    latent_dim=11,
                                                    hidden_neus=50)
                    
                    clients = []
                    for i, client in enumerate(client_info):
                        client_trainer = ClientTrainer(
                            model=copy.deepcopy(global_model),
                            train_loader=client["train_loader"],
                            save_dir=client['save_dir'],
                            epoch=epoch,
                            lr_rate=lr_rate,
                            update_type=update_type,
                            client_id=i+1,
                            peers=[]
                        )
                        clients.append(client_trainer)
                    
                    # Update peers for each client
                    for client_trainer in clients:
                        client_trainer.peers = [c for c in clients if c.client_id != client_trainer.client_id]
                    
                    results = []
                    client_latent = {}
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    for round in range(num_rounds):
                        logging.info(f"Starting round {round+1}")
                        client_latent[round] = {}
                        
                        selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                        selected_clients = [client_info[i] for i in selected_idx]
                        selected_trainers = [clients[i] for i in selected_idx]
                        selected_client_ids = [trainer.client_id for trainer in selected_trainers]
                        logging.info(f"Selected clients for training in round {round+1}: {selected_client_ids}")
                        
                        # Train locally
                        for i, (client, trainer) in enumerate(zip(selected_clients, selected_trainers)):
                            logging.info(f"Training client {trainer.client_id}...")
                            trainer.run(client["train_loader"], client["valid_loader"])
                        
                        # Compute MSE scores for voting after local training
                        mse_scores = []
                        for i, (client, trainer) in enumerate(zip(selected_clients, selected_trainers)):
                            mse_score = compute_mse_score(trainer.model, client["valid_loader"], device, trainer.client_id)
                            if mse_score > 1e6:
                                logging.warning(f"Client {trainer.client_id} has abnormally high MSE score: {mse_score}. Excluding from voting.")
                                continue
                            mse_scores.append((i, mse_score))
                            logging.info(f"Client {trainer.client_id} MSE score: {mse_score}")
                        
                        # Select aggregator based on lowest MSE score
                        if not mse_scores:
                            logging.error("No valid MSE scores for voting. Selecting random aggregator.")
                            aggregator_idx = random.randint(0, len(selected_trainers) - 1)
                        else:
                            aggregator_idx = min(mse_scores, key=lambda x: x[1])[0]
                        aggregator_trainer = selected_trainers[aggregator_idx]
                        aggregator_client = selected_clients[aggregator_idx]
                        logging.info(f"Selected client {aggregator_trainer.client_id} as aggregator for round {round+1}")
                        
                        # Aggregator performs model aggregation
                        logging.info(f"Client {aggregator_trainer.client_id} aggregating models...")
                        aggregator_trainer.aggregate_models(aggregator_client["valid_loader"], aggregator_trainer.receive_models(selected_client_ids=selected_client_ids))
                        
                        # Distribute aggregated model to all clients with validation
                        aggregated_model = copy.deepcopy(aggregator_trainer.model.state_dict())
                        all_client_ids = [trainer.client_id for trainer in clients]
                        accepted_clients = []
                        logging.info(f"Distributing aggregated model to all clients: {all_client_ids}")
                        for trainer, client in zip(clients, client_info):
                            if trainer.validate_model(aggregated_model, client["valid_loader"], threshold=0.1):
                                trainer.model.load_state_dict(aggregated_model)
                                accepted_clients.append(trainer.client_id)
                            else:
                                logging.info(f"Client {trainer.client_id} kept current model after validation.")
                        logging.info(f"Clients accepting aggregated model in round {round+1}: {accepted_clients}")
                        
                        # Evaluate global model
                        evaluator = Evaluator(aggregator_trainer.model, metric=metric, model_type=model_type)
                        round_results = {}
                        global_loss = 0.0
                        
                        with torch.no_grad():
                            for batch_input in aggregator_client["valid_loader"]:
                                _, _, loss = aggregator_trainer.model(batch_input[0].to(device))
                                global_loss += loss.item()
                            global_loss /= len(aggregator_client["valid_loader"])
                        
                        for i, client in enumerate(client_info):
                            logging.info(f"Evaluating client {i+1} - name: {client['device']}")
                            auc_score, test_latent, test_label = evaluator.evaluate(client["test_loader"], client["train_loader"])
                            round_results[client['device']] = auc_score
                            client_latent[round][client['device']] = (test_latent, test_label)
                        
                        round_results["global_loss"] = global_loss
                        round_results['join_clients'] = selected_client_ids
                        round_results = {f'round_{round+1}': round_results}
                        
                        with open(filename, 'a') as f:
                            f.write(json.dumps(round_results) + '\n')
                        
                        if global_loss < min_val_loss:
                            min_val_loss = global_loss
                            global_worse = 0
                        else:
                            global_worse += 1
                            if global_worse > global_patience:
                                logging.info("Early stopping in global round!")
                                break
                    
                    file_path = f'Checkpoint/LatentData/{network_size}/{no_Exp}/Run_{run}/latent_{model_type}_{update_type}.pkl'
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'wb') as f:
                        pickle.dump(client_latent, f)
                
                if model_type == "autoencoder":
                    global_model = Autoencoder(input_dim=dim_features,
                                            output_dim=dim_features,
                                            latent_dim=11,
                                            hidden_neus=50)
                    
                    clients = []
                    for i, client in enumerate(client_info):
                        client_trainer = ClientTrainer(
                            model=copy.deepcopy(global_model),
                            train_loader=client["train_loader"],
                            save_dir=client['save_dir'],
                            epoch=epoch,
                            lr_rate=lr_rate,
                            update_type=update_type,
                            client_id=i+1,
                            peers=[]
                        )
                        clients.append(client_trainer)
                    
                    # Update peers for each client
                    for client_trainer in clients:
                        client_trainer.peers = [c for c in clients if c.client_id != client_trainer.client_id]
                    
                    results = []
                    client_latent = {}
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    for round in range(num_rounds):
                        logging.info(f"Starting round {round+1}")
                        client_latent[round] = {}
                        
                        selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                        selected_clients = [client_info[i] for i in selected_idx]
                        selected_trainers = [clients[i] for i in selected_idx]
                        selected_client_ids = [trainer.client_id for trainer in selected_trainers]
                        logging.info(f"Selected clients for training in round {round+1}: {selected_client_ids}")
                        
                        # Train locally
                        for i, (client, trainer) in enumerate(zip(selected_clients, selected_trainers)):
                            logging.info(f"Training client {trainer.client_id}...")
                            trainer.run(client["train_loader"], client["valid_loader"])
                        
                        # Compute MSE scores for voting after local training
                        mse_scores = []
                        for i, (client, trainer) in enumerate(zip(selected_clients, selected_trainers)):
                            mse_score = compute_mse_score(trainer.model, client["valid_loader"], device, trainer.client_id)
                            if mse_score > 1e6:
                                logging.warning(f"Client {trainer.client_id} has abnormally high MSE score: {mse_score}. Excluding from voting.")
                                continue
                            mse_scores.append((i, mse_score))
                            logging.info(f"Client {trainer.client_id} MSE score: {mse_score}")
                        
                        # Select aggregator based on lowest MSE score
                        if not mse_scores:
                            logging.error("No valid MSE scores for voting. Selecting random aggregator.")
                            aggregator_idx = random.randint(0, len(selected_trainers) - 1)
                        else:
                            aggregator_idx = min(mse_scores, key=lambda x: x[1])[0]
                        aggregator_trainer = selected_trainers[aggregator_idx]
                        aggregator_client = selected_clients[aggregator_idx]
                        logging.info(f"Selected client {aggregator_trainer.client_id} as aggregator for round {round+1}")
                        
                        # Aggregator performs model aggregation
                        logging.info(f"Client {aggregator_trainer.client_id} aggregating models...")
                        aggregator_trainer.aggregate_models(aggregator_client["valid_loader"], aggregator_trainer.receive_models(selected_client_ids=selected_client_ids))
                        
                        # Distribute aggregated model to all clients with validation
                        aggregated_model = copy.deepcopy(aggregator_trainer.model.state_dict())
                        all_client_ids = [trainer.client_id for trainer in clients]
                        accepted_clients = []
                        logging.info(f"Distributing aggregated model to all clients: {all_client_ids}")
                        for trainer, client in zip(clients, client_info):
                            if trainer.validate_model(aggregated_model, client["valid_loader"], threshold=0.1):
                                trainer.model.load_state_dict(aggregated_model)
                                accepted_clients.append(trainer.client_id)
                            else:
                                logging.info(f"Client {trainer.client_id} kept current model after validation.")
                        logging.info(f"Clients accepting aggregated model in round {round+1}: {accepted_clients}")
                        
                        # Evaluate global model
                        evaluator = Evaluator(aggregator_trainer.model, metric=metric, model_type=model_type)
                        round_results = {}
                        global_loss = 0.0
                        
                        with torch.no_grad():
                            for batch_input in aggregator_client["valid_loader"]:
                                _, _, loss = aggregator_trainer.model(batch_input[0].to(device))
                                global_loss += loss.item()
                            global_loss /= len(aggregator_client["valid_loader"])
                        
                        for i, client in enumerate(client_info):
                            logging.info(f"Evaluating client {i+1} - name: {client['device']}")
                            auc_score = evaluator.evaluate(client["test_loader"], client["train_loader"])
                            round_results[client['device']] = auc_score
                            client_latent[round][client['device']] = (test_latent, test_label)
                        
                        round_results["global_loss"] = global_loss
                        round_results['join_clients'] = selected_client_ids
                        round_results = {f'round_{round+1}': round_results}
                        
                        with open(filename, 'a') as f:
                            f.write(json.dumps(round_results) + '\n')
                        
                        if global_loss < min_val_loss:
                            min_val_loss = global_loss
                            global_worse = 0
                        else:
                            global_worse += 1
                            if global_worse > global_patience:
                                logging.info("Early stopping in global round!")
                                break
                    
                    file_path = f'Checkpoint/LatentData/{network_size}/{no_Exp}/Run_{run}/latent_{model_type}_{update_type}.pkl'
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'wb') as f:
                        pickle.dump(client_latent, f)
                    
                    # file_path = f'Checkpoint/LatentData/{network_size}/{no_Exp}/Run_{run}/latent_{model_type}_{update_type}.pkl'
                    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    # with open(file_path, 'wb') as f:
                    #     pickle.dump(client_latent, f)
