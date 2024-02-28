# %%
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, f1_score
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset

#  %%
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-shots", "--shots", help = "Number of Shots")
parser.add_argument("-chunks", "--chunks", help = "Number of chunks")

 
# Read arguments from command line
args = parser.parse_args()
# %%
print("loading pickle")
import joblib
data = []
# chunks = int(args.chunks)
# numberofShots = int(args.shots)
chunks = 25
numberofShots = 0
for i in tqdm(range(chunks)):
    with open(f'/home//probing/llama2_embedding_nonShuffle/{numberofShots}shot/chunks/chunk_{i}.pickle', 'rb') as f:
        data.extend(pickle.load(f))
print("pickle loaded")
# breakpoint()
# %%
# %%
positive_elements = [[] for _ in range(33)]
negative_elements = [[] for _ in range(33)]
unrelated_elements = [[] for _ in range(33)]
import random
for d in data:
    for key in d.keys():
        if('first_first' in key):
            if('positive' in key):
                for layer in range(33):
                    positive_elements[layer].extend(d[key][layer].numpy())
            if('negative' in key):
                for layer in range(33):
                    negative_elements[layer].extend(d[key][layer].numpy())
            if('unrelated' in key):
                for layer in range(33):
                    unrelated_elements[layer].extend(d[key][layer].numpy())
# %%
# for layer in range(33):
#     random.shuffle(positive_elements[layer])
#     random.shuffle(negative_elements[layer])
#     random.shuffle(unrelated_elements[layer])
#     if(len(unrelated_elements[layer]) > 3 * len(positive_elements[layer])):
#         import random
#         # random.shuffle(unrelated_elements[layer])
#         unrelated_elements[layer] = unrelated_elements[layer][:3*len(positive_elements[layer])]

            

# %%
def getEmbeddingAndLabel(layer):
    positive_element_layer = positive_elements[layer]
    negative_element_layer = negative_elements[layer]
    unrelated_element_layer = unrelated_elements[layer]

    positive_labels = np.ones((len(positive_element_layer), 1))
    negative_labels = np.zeros((len(negative_element_layer), 1))
    unrelated_labels = np.ones((len(unrelated_element_layer), 1)) * (2)
    # print(len(positive_labels), len(negative_labels))
    X = np.vstack((positive_element_layer, negative_element_layer, unrelated_element_layer))
    y = np.vstack((positive_labels, negative_labels, unrelated_labels)).flatten()

    return X,y

# %%
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        # self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.relu1(out)
        # out = self.fc2(out)
        # out = self.relu2(out)
        # out = self.fc3(out)
        # out = self.relu3(out)
        # out = self.fc4(out)
        return out



# %%
def trainMLP(X_train, y_train):

    input_size = X_train.shape[1]
    hidden_size1 = 128
    hidden_size2 = 64
    hidden_size3 = 32
    num_classes = len(set(y_train))

    model = MLPClassifier(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Early stopping parameters
    patience = 10  # Number of epochs with no improvement to wait before stopping
    best_loss = float('inf')
    counter = 0
    batch_size = int(len(X_train) / 60)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 20
    loss_array = []
    with tqdm(total=num_epochs, desc="Training Progress") as pbar:

        for epoch in range(num_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                # Forward pass
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss_array.append(loss)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                del batch_x, batch_y, outputs, loss

            average_loss = total_loss / len(train_loader)


            pbar.set_postfix({"Loss": average_loss})
            pbar.update(1)

            # if (epoch + 1) % 10 == 0:
                # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            # Early stopping logic
            if average_loss < best_loss:
                best_loss = average_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch+1} (Best Validation Loss: {best_loss:.4f})')
                    break



    # clf = MLPClassifier(hidden_layer_sizes=(128,64), random_state=42, early_stopping=True, verbose=True)
    # clf.fit(X_train, y_train)

    return model, loss_array

def predict(model, X_test, y_test):
    with torch.no_grad():
        X_test = X_test.to(device)
        test_outputs = model(X_test)
        del X_test
        _, predicted = torch.max(test_outputs.data, 1)
        y_test_numpy = y_test.cpu().numpy()
        predicted_cpu = predicted.cpu().numpy()
        accuracy = accuracy_score(y_test_numpy, predicted_cpu)
        f1_weighted = f1_score(y_test_numpy, predicted_cpu, average='weighted')  # Calculate weighted F1 score
        f1_macro = f1_score(y_test_numpy, predicted_cpu, average='macro')  # Calculate macro F1 score
        f1_micro = f1_score(y_test_numpy, predicted_cpu, average='micro')  # Calculate micro F1 score
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Weighted F1 Score: {f1_macro:.4f}')
        report = classification_report(y_test_numpy, predicted_cpu)
        print("Classification Report:")
        print(report)
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.2f}")

    # f1_weighted = f1_score(y_test, y_pred, average='weighted')
    # f1_macro = f1_score(y_test, y_pred, average='macro')
    # f1_micro = f1_score(y_test, y_pred, average='micro')
    # print(f"F1-score: {f1:.2f}")
    # Print classification report
    
    return accuracy, f1_weighted, f1_macro, f1_micro

# %%
# X,y = getEmbeddingAndLabel(0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train = X_train.reshape((X_train.shape[0], -1))
# X_test = X_test.reshape((X_test.shape[0], -1))

# %%
# X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
# y_train = torch.tensor(y_train, dtype=torch.long).to(device)
# X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# %%

# %%
accuracy_values = []
f1_values = []
f1_weighted_values = []
losses = []
from tqdm import tqdm
for layer in tqdm(range(33)):
    X, y = getEmbeddingAndLabel(layer)
    # print(len(X), len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42,shuffle=False)
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    print(f"length of train data: {len(X_train)}")
    print(f"length of test data: {len(X_test)}")
    # breakpoint()
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model, loss_array = trainMLP(X_train, y_train)
    print(f"Layer {layer}")
    accuracy, f1_weighted, f1_macro, f1_micro = predict(model, X_test, y_test)
    accuracy_values.append(accuracy)
    f1_values.append((f1_weighted, f1_macro, f1_micro))
    f1_weighted_values.append(f1_weighted)
    losses.append(loss_array)
    with open(f'f1_values{numberofShots}_first_first.pickle', 'wb') as f:
        pickle.dump(f1_values, f)
    with open(f'loss_array_{numberofShots}_first_first.pickle', 'wb') as f:
        pickle.dump(losses, f)
    
    del X_train, X_test, y_train, y_test, model
    torch.cuda.empty_cache()

# %%
with open(f'f1_values_llama2_all_nonShuffle_{numberofShots}_first_first.pickle', 'wb') as f:
    pickle.dump(f1_values, f)

# %%
import matplotlib.pyplot as plt

plt.plot(accuracy_values, marker='o', linestyle='-', color='b', label='Data')
plt.xlabel('Layer Number')
plt.ylabel('Accuracy')
plt.title(f'Number of positive samples={len(positive_elements[0])}, Number of Negative samples={len(negative_elements[0])} first_first llama2')
plt.legend()
plt.grid(True)

plt.savefig(f'accuracy for pair_probing_task all non shuffle first_first with {numberofShots} shot llama2')
plt.clf()

# %%
import matplotlib.pyplot as plt

plt.plot(f1_weighted_values, marker='o', linestyle='-', color='b', label='Data')
plt.xlabel('Layer Number')
plt.ylabel('F1')
plt.title(f'Number of positive samples={len(positive_elements[0])}, Number of Negative samples={len(negative_elements[0])} first_first llama2')
plt.legend()
plt.grid(True)

plt.savefig(f'f1 score for pair_probing_task all nonShuffle first_first with {numberofShots} shot llama2')
plt.clf()

# %%


# %%
# import pickle

# # Size of each chunk (number of items)
# chunk_size = 100000

# # Open the original pickle file for reading
# with open('/home//probing/vicunaFinetuned/0shot/pairsInfo_full.pickle', 'rb') as f:
#     part_number = 0
#     while True:
#         chunk = []
#         try:
#             # Read a chunk of data from the pickle file
#             for _ in range(chunk_size):
#                 item = pickle.load(f)
#                 chunk.append(item)
#         except EOFError:
#             # End of file reached
#             break

#         # Save the chunk as a separate pickle file
#         with open(f'chunks/part_{part_number}.pickle', 'wb') as part_file:
#             pickle.dump(chunk, part_file)
        
#         part_number += 1
#         print(part_number)

# %%



