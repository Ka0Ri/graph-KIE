import os
import torch
import torch.nn.functional as F
from model import InvoiceGCN
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import sys

def load_train_test_split(save_fd):
    # import pdb; pdb.set_trace()
    train_data = torch.load(os.path.join(save_fd, 'train_data.dataset'))
    test_data = torch.load(os.path.join(save_fd, 'test_data.dataset'))
    return train_data, test_data


train_data, test_data = load_train_test_split(save_fd='dataset/processed')
print(train_data)
print(test_data)

model = InvoiceGCN(input_dim=train_data.x.shape[1], chebnet=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.9)

train_data = train_data.to(device)
test_data = test_data.to(device)
indices = torch.where((train_data.y < 0) & (train_data.y < 5))
train_data.y[indices] = 0
# indices = torch.where((train_data.y < 0) & (train_data.y < 5))
# train_data.y[indices] = 0

# estimate class weights for imbalanced data
_class_weights = compute_class_weight(class_weight='balanced',
                                      classes=train_data.y.unique().cpu().numpy(),
                                      y=train_data.y.cpu().numpy())

print(_class_weights)

epochs = 200
for epoch in range(0, epochs):
    model.train()
    optimizer.zero_grad()

    
    # used to convert the class labels from 1-based indexing to 0-based indexing.
    loss = F.nll_loss(model(train_data), train_data.y - 1,
                        weight=torch.FloatTensor(_class_weights).to(device))
    loss.backward()
    optimizer.step()

    # calculate accuracy on 5 classes
    with torch.no_grad():
        if epoch % 10 == 0:
            model.eval()

            # forward model
            for index, name in enumerate(['train', 'test']):
                _data = eval(f'{name}_data')
                y_pred = model(_data).max(dim=1)[1]
                y_true = (_data.y - 1)
                acc = y_pred.eq(y_true).sum().item() / y_pred.shape[0]

                y_pred = y_pred.cpu().numpy()
                y_true = y_true.cpu().numpy()
                print(f'{name} accuracy: {acc}')


                # confusion matrix metric:
                if name == 'test':
                    cm = confusion_matrix(y_true, y_pred)
                    class_acc = cm.diagonal() / cm.sum(axis=1)
                    print(classification_report(y_true, y_pred))

            loss_val = F.nll_loss(model(test_data), test_data.y - 1)
            print(f'Epoch: {epoch:3d}, train_loss: {loss:.4f}, val_loss: {loss_val:.4f}')
            print('>' * 50)
