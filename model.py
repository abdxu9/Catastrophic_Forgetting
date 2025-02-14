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

class Model(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.train_loader = None
        self.val_loader = None
        
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
        self.softmax = Softmax()
        
    def to(self, device):
        """
        Parameters
        ----------
        device : TYPE
            Change le matériel sur lequel notre model va opérer
        Returns
        -------
        None.

        """
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
        if not test:
            targets = np.concatenate(self.target)
            preds = np.concatenate(self.pred)
        else:
            targets = np.concatenate(self.test_target)
            preds = np.concatenate(self.test_pred)
    
        # Conversion des prédictions en classes 0 ou 1 (seuil de 0.5)
        bin_preds = (preds > 0.5).astype(int)
    
        return {
            'f1_score': round(f1_score(targets, bin_preds), 3),
            'accuracy': round(accuracy_score(targets, bin_preds), 3),
            'roc_auc': round(roc_auc_score(targets, preds), 3)
        }
    

    def plot_confusion_matrix(self, test=False):
        """
        Affiche la matrice de confusion pour les prédictions du modèle.
        Peut être utilisée pour les données d'entraînement, de validation ou de test.
        """
        if not test:
            targets = np.concatenate(self.target)
            preds = np.concatenate(self.pred)
        else:
            targets = np.concatenate(self.test_target)
            preds = np.concatenate(self.test_pred)

        preds = (preds > 0.5).astype(int)

        # Calcul de la matrice de confusion
        cm = confusion_matrix(targets, preds)

        # Affichage de la matrice de confusion
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
        plt.title("Matrice de Confusion")
        plt.xlabel("Prédictions")
        plt.ylabel("Réel")
        plt.show()

    
    def _make_train_step(self):
        def perform_train_step(x, y):
            self.model.train()
            output = self.model(
                input_ids=x['input_ids'],
                attention_mask=x['attention_mask'],
                token_type_ids=x['token_type_ids']
            )

            preds = self.sigmoid(output).detach().cpu().numpy()
            preds = (preds > 0.5).astype(int)  # Transformation en 0 ou 1
            preds = np.concatenate(preds)
            
            self.pred.append(preds)  # Sauvegarder les prédictions binaires
            self.target.append(y.detach().cpu().numpy())  # Sauvegarder les étiquettes réelles

            
            # Calcul de la perte
            loss = self.loss_fn(output.squeeze(-1), y)
            loss.backward()
    
            self.optimizer.step()
            self.optimizer.zero_grad()
    
            # Calcul de l'accuracy et renvoi
            accuracy = accuracy_score(np.concatenate(self.target), np.concatenate(self.pred))
            
            return [loss.item(), accuracy]
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
    
    def train(self, n_epochs, seed = 42):
        self.set_seed(seed)
        loop = tqdm(range(n_epochs), total=n_epochs, desc=f'Epoch {self.total_epochs}/{n_epochs}', 
                leave=False, dynamic_ncols=True, position=0)
        loop.refresh()
        for _ in loop:
            self.total_epochs += 1
            self.pred, self.target = [], []
            loss, accuracy = self._mini_batch(validation=False)
            self.losses.append(loss)
            with torch.no_grad():
                val_loss, val_accuracy=self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            loop.set_description(f'Epoch {self.total_epochs}')
            loop.set_postfix({'loss' : loss, 
                              'accuracy' : accuracy, 
                              'val_loss' : val_loss,
                              'val_accuracy' : val_accuracy}) 
            
    def predict(self, test_loader):
        self.model.eval()
        self.test_pred = []
        self.test_target = []
        with torch.no_grad():
            for batch in tqdm(test_loader, leave=True, dynamic_ncols=True):
                inputs = {
                    'input_ids': torch.as_tensor(batch['ids']).long().to(self.device),
                    'attention_mask': torch.as_tensor(batch['attention_mask']).long().to(self.device),
                    'token_type_ids': torch.as_tensor(batch['token_type_ids']).long().to(self.device)
                }
                output = self.model(**inputs)
                y = torch.as_tensor(batch['target']).to(self.device)
                
                preds = self.sigmoid(output).detach().cpu().numpy()
                preds = np.concatenate(preds)
                
                
                self.test_pred.append(preds)  # Sauvegarder les prédictions binaires
                self.test_target.append(y.detach().cpu().numpy())  # Sauvegarder les étiquettes réelles

        print(self.metrics(test=True))
    
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
        # Calculer les métriques à la fin de l'entraînement ou validation
        metrics = self.metrics(test=False)  # ou `self.metrics(test=True)` pour les métriques sur le test
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'val_losses': self.val_losses,
            'metrics': metrics  # Sauvegarder les métriques
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.to(self.device))  # Charger le fichier sur le bon device
        
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Charger les poids du modèle
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Charger l'optimiseur
        self.total_epochs = checkpoint['epoch']  # Restaurer le nombre d'époques
    
        print(f"✅ Modèle chargé depuis {filename}, entraîné pendant {self.total_epochs} époques.")


