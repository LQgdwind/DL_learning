import numpy as np
import torch.nn
import torch
import matplotlib.pyplot as plt

data_set = np.loadtxt("../data/german.data-numeric")

row, col = data_set.shape
print("row = {}".format(row))
for i in range(col - 1):
    mean_val = np.mean(data_set[:, i])
    std_val = np.std(data_set[:, i])
    data_set[:, i] = (data_set[:, i]-mean_val)/std_val

# 归一化处理(每一列都进行了归一化)

np.random.shuffle(data_set)
# 打乱数据

train_set = data_set[0:900, :col-1]
print(train_set.shape)
train_label = data_set[:900, col-1]-1
print(train_label.shape)
test_set = data_set[900:, :col-1]
test_label = data_set[900:, col-1]-1


class LogisticModel(torch.nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear_layer = torch.nn.Linear(in_features=24,
                                            out_features=2)

    def forward(self, input_tensor):
        output = torch.sigmoid(self.linear_layer(input_tensor))
        return output

    def __call__(self, input_tensor):
        return self.forward(input_tensor=input_tensor)

net = LogisticModel()
loss = torch.nn.CrossEntropyLoss()
epochs = 2000
trainer = torch.optim.SGD(net.parameters(), lr=0.2)
accuracy_test_list = []
accuracy_train_list = []
loss_list = []
for epoch in range(epochs-1):
    net.train()
    input_tensor = torch.from_numpy(train_set).float()
    label_tensor = torch.from_numpy(train_label).long()
    output_tensor = net(input_tensor)
    l = loss(output_tensor, label_tensor)
    trainer.zero_grad()
    l.backward()
    trainer.step()
    if epoch % 40 == 0:
        tensor_predict = net(torch.from_numpy(test_set).float())
        # print(tensor_predict.size())
        # print(test_label.shape)
        tmp = 0.
        for i in range(100-1):
            if tensor_predict[i][0]>tensor_predict[i][1] and test_label[i] == 0:
                tmp=tmp+1
            elif tensor_predict[i][0]<tensor_predict[i][1] and test_label[i] == 1:
                tmp=tmp+1
        tmp2 = 0.
        for i in range(900-1):
            if output_tensor[i][0]>output_tensor[i][1] and train_label[i] == 0:
                tmp2=tmp2+1
            elif output_tensor[i][0]<output_tensor[i][1] and train_label[i] == 1:
                tmp2=tmp2+1
        accuracy_test_list.append(tmp/100)
        accuracy_train_list.append(tmp2/900)
        loss_list.append(l.item())
plt.plot(np.linspace(0,2000,50,endpoint=True),np.array(accuracy_test_list),
             'r-',
             label="epochs and accuracy_rate on test_set")
plt.plot(np.linspace(0,2000,50,endpoint=True),np.array(accuracy_train_list),
             'b-',
             label="epochs and accuracy_rate on train_set")
plt.plot(np.linspace(0,2000,50,endpoint=True),np.array(loss_list),
             'g-',
             label="Cross_Entropy_loss")

plt.ylim(0.4, 0.9)
plt.ylabel("accuracy_rate")
plt.xlabel("epochs")
plt.legend()
plt.title("logistic_regression_training")
plt.show()
