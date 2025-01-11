import torch
from torch import nn
from pandas import DataFrame
import seaborn as sn
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np


def train_step(model, data_loader, loss_fn, optimizer, device):
    """ one forward and backwards pass """
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    accum_loss = 0
    correct_predictions = 0
    total_samples = 0

    #mx = -1e9
    #mn = 1e9
    for batch, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        # Compute prediction and loss
        y_pred = model(X) #.squeeze()
        loss = loss_fn(y_pred, y)
        # Backpropagation
        loss.backward()
        #nn.utils.clip_grad_value_(model.parameters(), clip_value=0.3)
        #mx = max(max([p.grad.max().item() for p in model.parameters()]), mx)
        #mn = min(min([p.grad.min().item() for p in model.parameters()]), mn)
        optimizer.step()

        accum_loss += loss.item()

        #accuracy = torch.cat((accuracy, (torch.argmax(y, dim=1) == torch.argmax(y_pred, dim=1))), dim=0)
        correct_predictions += torch.all(y == (nn.Sigmoid()(y_pred)>=0.5), dim=1).sum().item()
        total_samples += y.size(0)
    #print(f"MAX {mx}")
    #print(f"MIN {mn}")
    accuracy = correct_predictions / total_samples
    return accum_loss / len(data_loader), accuracy

def eval_step(model, data_loader, loss_fn, device):
    """ evaluates the model in the data given a loss """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    accum_loss = 0
    correct_predictions = 0
    total_samples = 0


    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):

            X = X.to(device)
            y = y.to(device)

            # Compute prediction
            y_pred = model(X)

            # Compute loss
            loss = loss_fn(y_pred, y)

            accum_loss += loss.item()
            
            #correct_predictions += (torch.argmax(y_pred, dim=1) == torch.argmax(y, dim=1)).sum().item()
            correct_predictions += torch.all(y == (nn.Sigmoid()(y_pred)>=0.5), dim=1).sum().item()
            total_samples += y.size(0)

    accuracy = correct_predictions / total_samples
    return accum_loss / len(data_loader), accuracy

# Dado un modelo entrenado y un dataset te da métricas de evaluar el modelo
def eval_multilabel(model, test_loader, actions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y = torch.tensor([])
    Yp = torch.tensor([])
    
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):

            Y = torch.cat((Y, y.cpu()), dim = 0)

            X = X.to(device)
            y = y.to(device)

            # Compute prediction
            y_pred = model(X)
            
            Yp = torch.cat((Yp, nn.Sigmoid()(y_pred.cpu())>=0.5), dim = 0)

    cm = multilabel_confusion_matrix(Y, Yp)

    for i, c in enumerate(cm):
        df_cm = DataFrame(c, index=['no', actions[i]], columns=['no', actions[i]])
        plt.figure(figsize=(10, 10))
        plt.title(actions[i])
        ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='g')
    
    summary = classification_report(Y, Yp, target_names=actions, zero_division=1.0)

    print(f"Accuracy: {torch.all(Y == Yp, dim=1).sum().item() / Y.shape[0] * 100:.2f}")
    print(summary)
    
    return Y, Yp
