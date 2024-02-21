import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import random
import time
from sklearn.model_selection import train_test_split

import joblib
import pickle
import matplotlib.pyplot as plt

#from eval_utils import *

from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import optuna
import torch.nn as nn

from thop import profile

from torchsummary import summary


torch.manual_seed(42)


tf.config.experimental_run_functions_eagerly(True)

def find_best_f1(pred, label, min_thd, max_thd, n_bins):
    f1_scores = []
    term = (max_thd - min_thd)/(n_bins-1)
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    for i in range(n_bins):
        pred_labels = put_labels(pred, min_thd + i*term)
        f1_scores.append(f1_score(label, pred_labels))
    
    max_id = f1_scores.index(max(f1_scores))

    if f1_scores[max(max_id-1, 0)] == f1_scores[max_id] == f1_scores[min(max_id+1, n_bins-1)]:
        return min_thd + max_id*term, f1_scores[max_id]
    else:
        return find_best_f1(pred, label, max(min_thd + max_id*term - term/2, min_thd), min(min_thd + max_id*term + term/2, max_thd), n_bins)

def put_labels(distance, threshold):
    distance = np.array(distance)
    threshold = np.array(threshold)  # Ensure threshold is a numpy array
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

best_f1 = 0  # Initialize best F1 score
best_model = None  # Initialize best model

def calc_p2p(predict, actual):
    tp = np.sum(predict * actual)
    tn = np.sum((1-predict) * (1-actual))
    fp = np.sum(predict * (1-actual))
    fn = np.sum((1-predict) * actual)
    
    precision = tp / (tp + fp + 0.000001)
    recall = tp / (tp + fn + 0.000001)
    f1 = 2 * precision * recall / (precision + recall + 0.000001)
    return f1, precision, recall, tp, tn, fp, fn

def get_trad_f1(score, label):
    score = np.asarray(score)
    maxx = float(score.max())
    minn = float(score.min())
    
    label = np.asarray(label)
    actual = label > 0.1
    
    grain = 1000
    max_f1 = 0.0
    max_f1_thres = 0.0
    p = 0
    r = 0
    for i in range(grain):
        thres = (maxx-minn)/grain * i + minn
        predict = score > thres
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_thres = thres
            p = precision
            r = recall
            
    return max_f1, max_f1_thres, p, r

    
    
Training_WADI_RAW = pd.read_csv("Dataset/WADI/WADI_train.csv") #"file can get to unzip WADI_train.zip"

TEST_WADI_RAW = pd.read_csv("Dataset/WADI/WADI_test.csv") #"file can get to unzip WADI_test.zip"


C_TEST_WADI_RAW=TEST_WADI_RAW.drop(['attack'], axis = 1)


MTS_cad_WADI_1 = pd.read_csv("Dataset/WADI/WADI_prediction_value/1_WADI_MTS_CAD_prediction_score.csv")
MTAD_gat_2 = pd.read_csv("Dataset/WADI/WADI_prediction_value/2_WADI_mtad_gat_prediction_score.csv")
GANF_3 = pd.read_csv("Dataset/WADI/WADI_prediction_value/3_WADI_ganf_prediction_score.csv")
ANOMALY_transformer_4 = pd.read_csv("Dataset/WADI/WADI_prediction_value/4_WADI_anomaly_transformer_prediction_score.csv")
RANSynCoder_5 = pd.read_csv("Dataset/WADI/WADI_prediction_value/5_WADI_RANSyn_prediction_score.csv")
Autoencoder_6 = pd.read_csv("Dataset/WADI/WADI_prediction_value/6_WADI_Autoencoder_prediction_score.csv")
USAD_7 = pd.read_csv("Dataset/WADI/WADI_prediction_value/7_WADI_USAD_prediction_score.csv")
GDN_8 = pd.read_csv("Dataset/WADI/WADI_prediction_value/8_WADI_GDN_w_prediction_scores.csv")
LSTM_9 = pd.read_csv("Dataset/WADI/WADI_prediction_value/9_WADI_lstm_prediction_score.csv")
MSCRED_10 =pd.read_csv("Dataset/WADI/WADI_prediction_value/10_WADI_mscred_prediction_score.csv")


list_WADI_model=[MTS_cad_WADI_1['score'],MTAD_gat_2['score'],GANF_3['score'],ANOMALY_transformer_4['score'],RANSynCoder_5['score'],Autoencoder_6['score'],USAD_7['score'],GDN_8['score'], LSTM_9['score'],MSCRED_10['score']] 


WADI_anomaly_score_concate = pd.concat((list_WADI_model[0], list_WADI_model[1], list_WADI_model[2], list_WADI_model[3], list_WADI_model[4], list_WADI_model[5], list_WADI_model[6], list_WADI_model[7], list_WADI_model[8], list_WADI_model[9]), axis = 1)


WADI_label=TEST_WADI_RAW['attack']





X_train, X_test, y_train, y_test = train_test_split(WADI_anomaly_score_concate, WADI_label, test_size=0.92,  random_state=1234)

C_X_train, C_X_test, C_y_train, C_y_test = train_test_split(C_TEST_WADI_RAW, WADI_label, test_size = 0.92, random_state=1234)



WADI_feature_score_concate = pd.concat((C_X_train,X_train), axis = 1)

WADI_feature_score_concate_valid = pd.concat((C_X_train,X_train), axis = 1)

WADI_feature_score_concate_test = pd.concat((C_X_test,X_test), axis = 1)


train_dataset = TensorDataset(torch.FloatTensor(WADI_feature_score_concate.values), torch.FloatTensor(y_train.values))

valid_dataset = TensorDataset(torch.FloatTensor(WADI_feature_score_concate_valid.values), torch.FloatTensor(y_train.values))

test_dataset = TensorDataset(torch.FloatTensor(WADI_feature_score_concate_test.values), torch.FloatTensor(C_y_test.values))


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, hidden_dim3, hidden_dim4, activation_fn_name):
        super(NeuralNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation_fn1 = self._get_activation_fn(activation_fn_name)
        #self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        #self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.activation_fn2 = self._get_activation_fn(activation_fn_name)
        #self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        #self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.activation_fn3 = self._get_activation_fn(activation_fn_name)
        #self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        #self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.activation_fn4 = self._get_activation_fn(activation_fn_name)
        #self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc5= nn.Linear(hidden_dim4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.activation_fn1(x)
        #x = self.dropout1(x)
        
        x = self.fc2(x)
        #x = self.bn2(x)
        x = self.activation_fn2(x)
        #x = self.dropout2(x)
        
        x = self.fc3(x)
        #x = self.bn3(x)
        x = self.activation_fn3(x)
        #x = self.dropout3(x)
        
        x = self.fc4(x)
        #x = self.bn3(x)
        x = self.activation_fn4(x)
        #x = self.dropout4(x)
        
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x

    def _get_activation_fn(self, name):
        
        if name == "ReLU":
            return nn.ReLU()
        elif name == "LeakyReLU":
            return nn.LeakyReLU()
        elif name == "Tanh":
            return nn.Tanh()
        elif name == "Sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {name}")



def get_test_f1(score, label,thres):
    score = np.asarray(score)
    maxx = float(score.max())
    minn = float(score.min())
    
    label = np.asarray(label)
    actual = label > 0.1
    
    grain = 1000
    max_f1 = 0.0
    max_f1_thres = 0.0
    p = 0
    r = 0
       
    predict = score > thres
    f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
    max_f1 = f1
    max_f1_thres = thres
    p = precision
    r = recall
            
    
    return max_f1, max_f1_thres, p, r


    
teacher_load_student = joblib.load('Savemodel/WADI/WADI_best_optuna.pkl')

df_teacher = teacher_load_student.trials_dataframe().drop(['number','datetime_start','datetime_complete','duration','state'], axis=1)



trial_num = 4

best_params = teacher_load_student.trials[trial_num].params


teacher_model = NeuralNet(input_dim=WADI_feature_score_concate.shape[1], 
                  hidden_dim=best_params["hidden_dim"], 
                  hidden_dim2=best_params["hidden_dim2"],
                  hidden_dim3=best_params["hidden_dim3"],
                  hidden_dim4=best_params["hidden_dim4"],
                  activation_fn_name=best_params["activation_fn"])
             
                  
teacher_model.load_state_dict(torch.load(f'Savemodel/WADI/Teacher_model_trial_{trial_num}.pth'))
teacher_model.eval()


input_data = torch.randn(1, 133)
summary(teacher_model, input_size=input_data)


input_tensor_meta = torch.tensor(WADI_feature_score_concate.iloc[0].to_numpy(), dtype=torch.float32)

meta_flops, meta_params = profile(teacher_model, inputs=(input_tensor_meta,))

print(f"meta-learner FLOPs: {meta_flops}, meta-learner Parameters: {meta_params}")


y_pred_values_valid=[]
y_true_valid=[]

with torch.no_grad():
    for data, target in valid_loader:  
        output_valid = teacher_model(data).squeeze()
        y_pred_values_valid.extend(output_valid.tolist())
        y_true_valid.extend(target.tolist())

y_pred_values_test = []
y_true_test = []


with torch.no_grad():
    for data, target in test_loader:
        output = teacher_model(data).squeeze()
        #y_pred_values_test.extend(output.tolist())
        y_pred_values_test.extend(output.flatten().tolist())
        batch_true_labels_test = [int(label) for label in target.tolist()]
        y_true_test.extend(batch_true_labels_test)


thresholds = np.linspace(0, 1, 100)
best_threshold = 0
max_f1 = 0
for thd in thresholds:
    y_pred = [1 if y > thd else 0 for y in y_pred_values_test]
    f1 = f1_score(y_true_test, y_pred, zero_division=1)
    if f1 > max_f1:
        max_f1 = f1
        best_threshold = thd


valid_f1,valid_treshold,_,_=get_trad_f1(y_pred_values_valid, y_true_valid)

test_f1,test_treshold,precision,recall=get_test_f1(y_pred_values_test, y_true_test,valid_treshold)


y_train_pred_values_pretrain_teacher = []
y_train_true = []

# Pre-trained teacher model, we put train dataset to get a predictive output
with torch.no_grad():
    for data, target in train_loader:
        output = teacher_model(data).squeeze()
        y_train_pred_values_pretrain_teacher.extend(output.tolist())
        batch_true_labels = [int(label) for label in target.tolist()]
        y_train_true.extend(batch_true_labels)

df = pd.DataFrame(y_train_pred_values_pretrain_teacher)

df.to_csv("WADI_pretrain_prediction.csv", index=False, header=False) 


from tensorflow import keras
import torch.optim as optim


import torch.nn.functional as F


def knowledge_distillation_loss(y_true, student_y_pred, teacher_preds_value, alpha, temperature): #0.5 1.0
    # Ensure that the student predictions have the same shape as the true labels
    student_y_pred = torch.squeeze(student_y_pred)

    # Cross-entropy loss
    ce_loss = F.binary_cross_entropy_with_logits(student_y_pred, y_true)#F.binary_cross_entropy_with_logits(student_y_pred, y_true)

    # Soften predictions and calculate distillation loss
    teacher_soft = torch.sigmoid(teacher_preds_value / temperature)
    student_soft = torch.sigmoid(student_y_pred / temperature)
    kd_loss = F.mse_loss(student_soft, teacher_soft)#F.binary_cross_entropy_with_logits(student_soft, teacher_soft) 

    # Combine losses
    combined_loss = (1 - alpha) * kd_loss + alpha * ce_loss
    return combined_loss


def train_on_batch(model, dataset_zip, optimizer, alpha, temp):
    total_loss = 0
    for (teacher_pred, X_train), true_label in dataset_zip:
        # Convert TensorFlow tensors to PyTorch tensors
        teacher_pred = torch.from_numpy(teacher_pred.numpy()).float()
        X_train = torch.from_numpy(X_train.numpy()).float()
        true_label = torch.from_numpy(true_label.numpy()).float()

        # Forward pass
        optimizer.zero_grad()  # Clear existing gradients
        student_y_pred = model(X_train)
        loss = knowledge_distillation_loss(true_label, student_y_pred, teacher_pred, alpha, temp)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(dataset_zip)


class StudentModel(torch.nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = torch.nn.Linear(123, 71)
        self.fc2 = torch.nn.Linear(71, 152)
        self.fc3 = torch.nn.Linear(152, 172)
        self.fc4 = torch.nn.Linear(172, 169)
        self.fc5 = torch.nn.Linear(169, 1)  

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return torch.sigmoid(self.fc5(x))  


student_model = StudentModel()  # Create an instance of the model


optimizer = optim.Adam(student_model.parameters(),lr=0.0001354368553070506)  # Pass the model instance  0.00001

print("student parameter")

input_data = torch.randn(1, 123)
summary(teacher_model, input_size=input_data)

epochs = 20
batch_size = 64

alpha_value = 0
temperature_value = 10


dataset_12 = tf.data.Dataset.from_tensor_slices((y_train_pred_values_pretrain_teacher, C_X_train))
dataset_label = tf.data.Dataset.from_tensor_slices(C_y_train)
dataset_zip = tf.data.Dataset.zip((dataset_12, dataset_label)).batch(batch_size)


for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
    print(f'Epoch {epoch + 1}/{epochs}')
    loss = train_on_batch(student_model, dataset_zip, optimizer, alpha=alpha_value, temp=temperature_value)
    print(f'Loss: {loss}')




student_model.eval()


#####use validation set to decide threshold 



with torch.no_grad():  # Disable gradient calculation
    y_predicted_valid = student_model(torch.tensor(C_X_train.to_numpy(), dtype=torch.float32))
    #y_predicted = student_model(torch.tensor(SWaT_feature_score_concate_test.to_numpy(), dtype=torch.float32))



y_predicted_np_valid = int(y_predicted_valid.numpy()) if y_predicted_valid.requires_grad else y_predicted_valid.detach().numpy()


predict_valid = y_predicted_np_valid.reshape(-1)
actual_valid = C_y_train.to_numpy().reshape(-1)


unique_values_valid_predict, counts_valid_predict = np.unique(y_predicted_np_valid, return_counts=True)


unique_values_ground_valid_actual, counts_ground_valid_actual = np.unique(actual_valid, return_counts=True)


valid_f1,valid_thresh,valid_p,valid_c = get_trad_f1(predict_valid,actual_valid)


input_tensor = torch.tensor(C_X_test.iloc[[0]].to_numpy(), dtype=torch.float32)
flops, params = profile(student_model, inputs=(input_tensor,))

print(f"FLOPs: {flops}, Parameters: {params}")




#### use test dataset

with torch.no_grad():  # Disable gradient calculation
    y_predicted = student_model(torch.tensor(C_X_test.to_numpy(), dtype=torch.float32))
    #y_predicted = student_model(torch.tensor(SWaT_feature_score_concate_test.to_numpy(), dtype=torch.float32))


y_predicted_np = int(y_predicted.numpy()) if y_predicted.requires_grad else y_predicted.detach().numpy()


predict = y_predicted_np.reshape(-1)
actual = C_y_test.to_numpy().reshape(-1)


unique_values, counts = np.unique(y_predicted_np, return_counts=True)


unique_values_ground, counts_ground = np.unique(actual, return_counts=True)


def get_trad_f1_final(score, label):
    score = np.asarray(score)
    maxx = float(score.max())
    minn = float(score.min())
    
    label = np.asarray(label)
    actual = label > 0.1
    
    predict = score > valid_thresh
    max_f1, p, r, tp, tn, fp, fn = calc_p2p(predict, actual)
    
    
    max_f1_thres= valid_thresh       
    print("Student model f1 score is %f and threshold is %f\n" %(max_f1, valid_thresh))
    return max_f1, max_f1_thres, p, r



print(get_trad_f1_final(predict,actual))