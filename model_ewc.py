# -*- coding: utf-8 -*-
"""
@author: Abdoulatuf COLO
"""
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
from torch.nn import Sigmoid, Softmax
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score

class ModelEWC(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.train_loader = None
        self.val_loader = None
        
        self.ewc_lambda = 0  # Facteur de pondération EWC
        self.fisher_information = None  # Matrice de Fisher
        self.optimal_params = None  # Poids sauvegardés après la tâche précédente
        
        self.seed = 42
        self.set_seed(self.seed)
        
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0
        
        self.target = []
        self.pred = []
        
        self.val_target = []
        self.val_pred = []
        
        self.test_target = []
        self.test_pred = []
        
        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()
        
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=1)
        
    def to(self, device):
        """Déplace le modèle vers l'appareil spécifié (CPU ou GPU)."""
        self.device = device
        self.model.to(self.device)
        
    def set_loader(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def metrics(self, test=False):
        """Calcul des métriques sur les prédictions du modèle."""
        if not test:
            targets = np.concatenate(self.target)
            preds = np.concatenate(self.pred)
        else:
            targets = np.concatenate(self.test_target)
            preds = np.concatenate(self.test_pred)
    
        bin_preds = (preds > 0.5).astype(int)  # Seuil de classification
    
        return {
            'f1_score': round(f1_score(targets, bin_preds), 3),
            'accuracy': round(accuracy_score(targets, bin_preds), 3),
            'roc_auc': round(roc_auc_score(targets, preds), 3)
        }

    def plot_confusion_matrix(self, test=False):
        """Affiche la matrice de confusion des prédictions."""
        if not test:
            targets = np.concatenate(self.target)
            preds = np.concatenate(self.pred)
        else:
            targets = np.concatenate(self.test_target)
            preds = np.concatenate(self.test_pred)

        preds = (preds > 0.5).astype(int)

        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
        plt.title("Matrice de Confusion")
        plt.xlabel("Prédictions")
        plt.ylabel("Réel")
        plt.show()

    def activate_ewc(self, ewc_lambda):
        """Active la régularisation EWC avec un facteur donné."""
        self.ewc_lambda = ewc_lambda

    def compute_fisher_information(self, dataloader):
        """Calcule la matrice de Fisher pour la tâche actuelle."""
        if self.ewc_lambda == 0:
            print("EWC désactivé, pas de calcul de Fisher")
            self.fisher_information = None  # On s'assure que la régularisation ne soit pas appliquée
            self.optimal_params = None
            return
        
        self.model.eval()
        fisher_information = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
    
        for batch in dataloader:
            x_batch = {
                'input_ids': batch['ids'].to(self.device),
                'token_type_ids': batch['token_type_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            }
            self.model.zero_grad()
            output = self.model(**x_batch)
            target_batch = batch['target'].to(self.device)
            loss = self.loss_fn(output.squeeze(-1), target_batch)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    fisher_information[name] += (param.grad ** 2)
    
        batch_count = len(dataloader)
        for name, fisher in fisher_information.items():
            fisher_information[name] = fisher / batch_count
    
        self.fisher_information = fisher_information
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        #print("Valeurs de Fisher après normalisation:")
        #for name, fisher in self.fisher_information.items():
            #print(f"{name}: {fisher.abs().max().item()}")
        return fisher_information
    
    def compute_ewc_loss(self, loss, verbose=0):
        """Ajoute la pénalité EWC à la perte de base."""
        if self.ewc_lambda > 0 and self.fisher_information is not None and self.optimal_params is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fisher_information:
                    fisher = self.fisher_information[name]
                    theta_old = self.optimal_params[name].to(param.device)
                    ewc_loss += torch.sum(fisher * (param - theta_old) ** 2)
            if verbose==1:
                print(f"EWC Loss avant pondération: {ewc_loss.item()}")
                loss += (self.ewc_lambda / 2) * ewc_loss
                print(f"EWC Loss après pondération: {(self.ewc_lambda / 2) * ewc_loss.item()}")
            else:
                loss += (self.ewc_lambda / 2) * ewc_loss
        return loss

    def _make_train_step(self):
        def perform_train_step(x, y):
            self.model.train()
            output = self.model(**x)
            preds = self.sigmoid(output).detach().cpu().numpy()
            self.pred.append((preds > 0.5).astype(int))
            self.target.append(y.detach().cpu().numpy())
    
            loss = self.loss_fn(output.squeeze(-1), y)
            #print(loss)
            loss = self.compute_ewc_loss(loss)
            #print(loss)
    
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
            return loss.item(), accuracy_score(np.concatenate(self.target), np.concatenate(self.pred))
        return perform_train_step
    
    
    
    def _make_val_step(self):
        def perform_val_step(x, y):
            self.model.eval()
            with torch.no_grad():
                output = self.model(
                    input_ids=x['input_ids'],
                    attention_mask=x['attention_mask'],
                    token_type_ids=x['token_type_ids']
                )
                # Appliquer le seuil pour obtenir des prédictions binaires
                preds = self.sigmoid(output).detach().cpu().numpy()
                preds = (preds > 0.5).astype(int)  # Transformation en 0 ou 1
                
                self.val_pred.append(preds)
                self.val_target.append(y.detach().cpu().numpy())
    
                # Calcul de la perte
                loss = self.loss_fn(output.squeeze(-1), y)
            return [loss.item(), accuracy_score(np.concatenate(self.val_target), np.concatenate(self.val_pred))]
        return perform_val_step
    
    def _mini_batch(self, validation=False):
        data_loader =self.val_loader if validation else self.train_loader
        step = self.val_step if validation else self.train_step
        
        if data_loader is None:
            return None
        
        mini_batch_losses = []
        mini_batch_accuracies = []
        for batch in data_loader:
            x_batch = {
                'input_ids': batch['ids'].to(self.device),
                'token_type_ids': batch['token_type_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            }
            
            target_batch = batch['target'].to(self.device)
            
            mini_batch_loss, mini_batch_accuracy = step(x_batch, target_batch)
            mini_batch_losses.append(mini_batch_loss)
            mini_batch_accuracies.append(mini_batch_accuracy)
            
        return [np.mean(mini_batch_losses), np.mean(mini_batch_accuracies)]

    def train(self, n_epochs, seed=42):
        self.set_seed(seed)
        loop = tqdm(range(n_epochs), desc=f'Epoch {self.total_epochs}/{n_epochs}', leave=False)
        
        for _ in loop:
            self.total_epochs += 1
            self.pred, self.target = [], []
            loss, accuracy = self._mini_batch()
            self.losses.append(loss)
            
            with torch.no_grad():
                val_loss, val_accuracy = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            
            loop.set_description(f'Epoch {self.total_epochs}')
            loop.set_postfix({'loss': loss, 'accuracy': accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

    def predict(self, test_loader):
        self.model.eval()
        self.test_pred, self.test_target = [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, leave=True):
                inputs = {
                    'input_ids': batch['ids'].to(self.device),
                    'token_type_ids': batch['token_type_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                output = self.model(**inputs)
                y = batch['target'].to(self.device)
                
                preds = self.sigmoid(output).detach().cpu().numpy()
                self.test_pred.append(preds)
                self.test_target.append(y.detach().cpu().numpy())

        return self.metrics(test=True)
        
    def plot_losses(self, save_path=None):
        plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        if save_path:
            plt.savefig(save_path)
    
   

    def save_checkpoint(self, filename):
        torch.save({
            'epoch': self.total_epochs,
            "ewc_lambda": self.ewc_lambda,
            'fisher_information': self.fisher_information,
            'optimal_params': self.optimal_params,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'val_losses': self.val_losses,
            'metrics': self.metrics()
        }, filename)
        
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.to(self.device))  # Charger le fichier sur le bon device
        
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Charger les poids du modèle
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Charger l'optimiseur
        self.total_epochs = checkpoint['epoch']  # Restaurer le nombre d'époques
    
        print(f"✅ Modèle chargé depuis {filename}, entraîné pendant {self.total_epochs} époques.")

