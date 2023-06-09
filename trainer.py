import torch # tensor operations
from torch import nn # package of layers and activation functions
from tqdm.auto import tqdm # progress bar
#from tqdm import tqdm # bar

class Trainer:

    def __init__(self, model_checkpoint=None, early_stopping=None):
        self.model_checkpoint = model_checkpoint
        self.early_stopping = early_stopping

    def classification_loss(self):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def compute_loss_metrics(self, batch, model, loss_fn, device):
        X, y = batch
        # Move to the device
        X = X.to(device)
        y = y.to(device)

        # Forward
        logits = model(X)

        # Compute loss
        loss = loss_fn(logits, y)

        # Calculate the accuracy
        acc = (logits.argmax(1) == y).type(torch.float).mean()
        
        return loss, acc


    def train_phase(self, train_dl, model, loss_fn, optimizer, device, pbar):
        '''
          Function that performs training for an epoch
      
          Args
            - dataloader: iterate over batches of data
            - model: model that receives an input and returns logits
            - loss_fn: loss function
            - optimizer: optimizer
            - device: device type responsible to load a tensor into memory
            - pbar: progress bar
        '''
        # Dataset size
        size = len(train_dl.dataset)

        # Train error
        train_loss, train_acc = 0., 0.

        model.train()
 
        for it, batch in enumerate(train_dl):
            X, _ = batch

            loss, acc = self.compute_loss_metrics(batch, model, loss_fn, device)

            # Accumulate the loss
            train_loss += loss.item() * X.shape[0]
            train_acc += acc.item() * X.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': acc})
            pbar.update(1)

        
        # Average per epoch
        train_loss /= size
        train_acc /= size
        
        # Update the results per epoch
        pbar.set_postfix({'loss': train_loss, 'acc': train_acc})

        return {"loss": train_loss, "acc": train_acc}


    def test_phase(self, test_dl, model, loss_fn, device):
        '''
          Function that performs testing for an epoch
      
          Args
            - dataloader: iterate over batches of data
            - model: model that receives an input and returns logits
            - loss_fn: loss function
            - device: device type responsible to load a tensor into memory
        '''
        # Dataset size
        size = len(test_dl.dataset)

        # Test error
        test_loss, test_acc = 0, 0

        model.eval()

        # Create progress bar
        pbar = tqdm(total=len(test_dl), desc=f'Validating',  position=0, leave=False)

        with torch.no_grad():
            for it, batch in enumerate(test_dl):
                X, _ = batch
                loss, acc = self.compute_loss_metrics(batch, model, loss_fn, device)
                # Accumulate the loss
                test_loss += loss.item() * X.shape[0]
                test_acc += acc.item() * X.shape[0]
          
                # Update the progress bar
                pbar.set_postfix({'loss': loss.item(), 'acc': acc})
                pbar.update(1)

        # Average per epoch
        test_loss /= size
        test_acc /= size
        
        # Update results per epoch
        pbar.set_postfix({'loss': test_loss, 'acc': test_acc})
        pbar.close()
        
        return {"loss": test_loss, "acc": test_acc}        


    def train(self, train_dl, val_dl, model, num_epochs, optimizer, device='cpu', scheduler=None):
        '''
          Function that performs training using train and val iterators
      
          Args
            - train_dl: training data iterator
            - val_dl: validation data iterator
            - model: model that receives an input and returns logits
            - num_epochs: number of epochs
            - optimizer: optimizer used to update the parameters
            - device:  device type responsible to load a tensor into memory
            - scheduler: scheduler used to decay the learning rate
        '''
        # Initialize the loss function
        loss_fn = self.classification_loss()

        model = model.to(device)

        # Results per epoch
        train_acc_history, train_loss_history = [], []
        val_acc_history, val_loss_history = [], []  

        # Iterate given a number of epochs
        for epoch in range(1, num_epochs+1):

            # Create progress bar that will summary results per epoch
            pbar = tqdm(total=len(train_dl), desc=f'Epoch {epoch}/{num_epochs}')
            
            train_results = self.train_phase(train_dl, model, loss_fn, optimizer, device, pbar)
            val_results = self.test_phase(val_dl, model, loss_fn, device)
            
            # Results
            train_loss, train_acc = train_results["loss"], train_results["acc"]
            val_loss, val_acc = val_results["loss"], val_results["acc"]    


            results = {'train_loss': train_loss, 'val_loss': val_loss, 
                       'train_acc': train_acc, 'val_acc': val_acc}
            # Scheduler
            if scheduler is not None:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
                results['lr'] = lr
            
            # Bar results
            pbar.set_postfix(results)
            pbar.close()


            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)     
        
            # Used in checkpoing and early stopping
            candidates = {'train_loss': train_loss, 'train_acc': train_acc,
                          'val_loss': val_loss, 'val_acc': val_acc}
            # Model checkpoint            
            if self.model_checkpoint is not None:                
                self.model_checkpoint(candidates, model)
            
            # Early stopping
            if self.early_stopping is not None and self.early_stopping.early_stop(candidates):
                print(f'Early stopped on epoch: {epoch}')
                break

        
        history = {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "train_acc": train_acc_history,
            "val_acc": val_acc_history
        }

        return history