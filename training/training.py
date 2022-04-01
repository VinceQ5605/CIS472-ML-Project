import torch
from torch import optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, train_data, LossFunction, optimizer):
    loss_history = []
    for (data, label) in tqdm(train_data, desc="iteration", unit="%", disable=True):
        # clear the gradients
        optimizer.zero_grad(set_to_none=True)
        
        # send the data and labels to the same device as the training model
        data = data.to(device)
#         print(label)
        label = label.to(device)

        # get the output from the model
        out = model(data)
        
        # calculate loss
        loss = LossFunction(out, label)
        
        # calculate gradients and perform gradient descent
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return loss_history
    
def evaluate(model, test_data):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for (data, label) in test_data:
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            answers = out.max(dim=1)[1]
#             answers = torch.max(out, dim=1)
#             print(answers)
            accuracy += (answers == label).sum()
    return accuracy