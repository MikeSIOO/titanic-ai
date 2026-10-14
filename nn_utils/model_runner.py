import numpy as np
import torch


class ModelRunner:

    def __init__(
            self,
            model: torch.nn.Sequential,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            loss_function: torch.nn.BCEWithLogitsLoss,
            optimizer: torch.optim.Optimizer,
            lr_scheduler,
            num_epochs: int
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs

    def train_model(self):
        loss_history = []
        train_history = []
        val_history = []

        for epoch in range(self.num_epochs):
            self.model.train()

            loss_accum = 0
            correct_samples = 0
            total_samples = 0

            for i_step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                prediction = self.model(x)
                loss_value = self.loss_function(prediction, y)

                self.optimizer.zero_grad()
                loss_value.backward()

                self.optimizer.step()

                loss_accum += loss_value.item()
                correct_samples += ((torch.sigmoid(prediction) > 0.5).float() == y).sum().item()
                total_samples += y.shape[0]

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            ave_loss = loss_accum / (i_step + 1)
            train_accuracy = float(correct_samples) / total_samples
            val_accuracy = self.compute_accuracy(self.model, self.val_loader)

            loss_history.append(float(ave_loss))
            train_history.append(train_accuracy)
            val_history.append(val_accuracy)

            print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

        return loss_history, train_history, val_history

    def evaluate_model(self):
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                prediction = self.model(x)
                predictions.append(prediction.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        return predictions

    def compute_accuracy(self, model, loader):
        model.eval()

        correct_samples = 0
        total_samples = 0

        with torch.no_grad():
            for i_step, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                prediction = model(x)

                correct_samples += ((torch.sigmoid(prediction) > 0.5).float() == y).sum().item()
                total_samples += y.shape[0]

        accuracy = float(correct_samples) / total_samples

        return accuracy
