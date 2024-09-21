import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from model import CustomModel


EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3


class FineTuningPipeline:
    def __init__(self, model_name, num_labels, bnb_config, pooling, device):
        self.device = device
        self.model = CustomModel(model_name, num_labels, bnb_config, pooling=pooling).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.freeze_base_model()
    
    def train(self, dataloader, val_dataloader, epochs=EPOCHS, patience=PATIENCE):
        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device).float()

                self.optimizer.zero_grad()

                logits = self.model(input_ids, attention_mask=attention_mask)

                loss = torch.nn.functional.cross_entropy(logits, labels)
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}")

            val_loss = self.evaluate(val_dataloader)
            print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), "custom_gte_model.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device).float()

                logits = self.model(input_ids, attention_mask=attention_mask)

                loss = torch.nn.functional.cross_entropy(logits, labels)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(dataloader)
        return avg_val_loss

    def predict(self, dataloader):
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inferencing"):
                input_ids, attention_mask, _ = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                logits = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                
                results.extend(probs)

        return results

    def get_trainable_params(self):
        return sum([param.requires_grad for param in self.model.parameters()])

    def freeze_base_model(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")