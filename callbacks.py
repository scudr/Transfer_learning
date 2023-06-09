import torch # tensor operations

# Model Checkpoint
class ModelCheckpoint:
    def __init__(self, path='checkpoint.pt', mode='min', monitor='val_loss', verbose=False):
        """
          Args:
              - path: path where the checkpoint will be saved/loaded
              - mode: whether to save the model with the min/max score given by monitor
              - monitor: score to compare according to the mode.
                         It can be val_loss, train_loss, val_acc, train_acc
              - verbose: whether to print messages when a new model is saved or not
        """
        self.path = path
        self.best_score = None
        self.verbose = verbose
        self.mode = mode
        self.monitor = monitor
        self.best_results = None

    def __call__(self, monitor_candidates, model):
        """
          Args:
              - monitor_candidates: dictionary with keys that can be accessed by
                                    monitor, the values of the dictionary are the
                                    score that will be compared
              - model: model with its parameters that will be saved or loaded

        """
        if self.monitor not in monitor_candidates:
            raise ValueError(f"Invalid monitor. Possible values: {monitor_candidates.keys()}") 
        score = monitor_candidates[self.monitor]

        if self.best_score is None or \
           (self.mode == 'min' and score < self.best_score) or \
           (self.mode == 'max' and score > self.best_score):
            if self.verbose:
                if self.best_score != None:
                    print(f"{self.monitor} changed ({self.best_score:.6f} --> {score:.6f}).  Saving model ...\n")
                else:
                    print(f"Saving model...\n")
            self.best_score = score
            self.best_results = monitor_candidates
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        
        
# Early Stopping
class EarlyStopping:
  
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, mode="min", verbose=False):
        """
          Args:
              - monitor: score to compare according to the mode.
                         It can be val_loss, train_loss, val_acc, train_acc
              - min_delta: minimum change in the monitored quantity that must 
                           be considered as an improvement, below which no 
                           further progress is made.
              - patience: number of epochs to wait before stopping the 
                          training process once the monitored quantity has 
                          stopped improving.
              - mode: whether to save the model with the min/max score given by monitor
              - verbose: whether to print messages when a new model is saved or not.
        """
        self.best_score = None
        self.verbose = verbose
        self.mode = mode
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

    def early_stop(self, monitor_candidates):
        """
          Args:
              - monitor_candidates: dictionary with keys that can be accessed by
                                    monitor, the values of the dictionary are the
                                    score that will be compared
              - model: model with its parameters that will be saved or loaded

        """
        if self.monitor not in monitor_candidates:
            raise ValueError(f"Invalid monitor. Possible values: {monitor_candidates.keys()}") 
        score = monitor_candidates[self.monitor]
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f"{self.monitor} improved. Best score: {self.best_score:.6f}\n")
        # score has to improve but at least min_delta
        elif(self.mode == 'min' and score > self.best_score - self.min_delta) or \
            (self.mode == 'max' and score < self.best_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"{self.monitor} improved. Best score: {self.best_score:.6f}\n")
          
        return False