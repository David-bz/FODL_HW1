from part1 import *
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
from collections import namedtuple
import pandas as pd
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
import time
from inception import *
import os

in_features= 3 * 32 * 32
num_classes = 10
selected = {'lr' : 0.00925,
            'momentum' : 0.9,
            'std' : 0.025}

class Part_2:
    def __init__(self, in_features, num_classes):
        torch.manual_seed(10)
        self.ds = HW1_Dataset(batch_size = 100, subset_portion = 0.05)
        self.in_features, self.num_classes =  in_features, num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dir = os.path.abspath('.')
        self.set_baseline_model()

    def perform_grid_search(self, subset_portion = 0.1):
        self.ds = HW1_Dataset(batch_size = 100, subset_portion = subset_portion)
        lr_values = [selected['lr']]
        momentum_values = [selected['momentum']]
        std_values = [selected['std']]
        res = []
        for lr in lr_values:
            for momentum in momentum_values:
                for std in std_values:
                    print(lr, momentum, std)
                    self.set_custom_model(lr, momentum, std)
                    train_res, test_res, training_time = self.train(num_epochs=150)
                    name = 'lr_{}_momentum_{}_std_{}'.format(lr, momentum, std)
                    self.plot_fig(name, name, train_res, test_res)
                    res.append([lr, momentum, std] + np.mean(train_res.accuracy[-3:]) + np.mean(train_res.loss[-3:]) +
                               np.mean(test_res.accuracy[-3:]) + np.mean(test_res.loss[-3:]) +
                               [training_time])
        with open("grid_res_6.pkl", "wb+") as f:
            pickle.dump(res, f)

    def set_baseline_model(self):
        # self.model = nn.Sequential(nn.Linear(self.in_features, 256),
        #                     nn.ReLU(),
        #                     nn.Linear(256, self.num_classes))
        self.model = inception_v3()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.008, momentum=0.9)

    def set_custom_model(self, lr=selected['lr'], mom=selected['momentum'], std=selected['std'], input_size = None,
                         hidden_dim = 256, layers=None):
        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.normal_(layer.weight, mean=0, std = std)
                layer.bias.data.fill_(0.02)
        in_features = self.in_features if input_size is None else input_size
        if layers is None:
            self.model = nn.Sequential(nn.Linear(in_features, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, self.num_classes))
        else:
            self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=mom)

    def set_dropout_model(self, dropout, weight_decay=3e-4, lr=selected['lr'], mom=selected['momentum'], std=selected['std']):
        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.normal_(layer.weight, mean=0, std = std)
                layer.bias.data.fill_(0.02)
        self.model = nn.Sequential(nn.Linear(in_features, 256),
                                   nn.Dropout(p = dropout),
                                   nn.ReLU(),
                                   nn.Linear(256, self.num_classes))
        self.model.apply(init_weights)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=mom, weight_decay=weight_decay)

    def set_optimization_experiment(self):
        self.set_custom_model(0.01, selected['momentum'], selected['std'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0025)
        train_res, test_res, train_time = self.train(num_epochs = 150)
        self.plot_fig('adam_optimizer_0.0025', 'adam_optimizer_0.001', train_res, test_res)

    def set_initialization_experiment(self):
        def apply_xavier(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.02)

        self.set_custom_model(selected['lr'], selected['momentum'], selected['std'])
        self.model.apply(apply_xavier)
        self.optimizer = optim.SGD(self.model.parameters(), lr=selected['lr'], momentum=selected['momentum'])
        train_res, test_res, train_time = self.train(num_epochs = 150)
        self.plot_fig('xavier_init_', 'xavier_init_', train_res, test_res)

    def set_pca_experiment(self, n_comp = 32*32):
        pca = sklearn.decomposition.PCA(n_comp, whiten=True)
        flat_train = self.ds.flatten(self.ds.dl_train)
        flat_test = self.ds.flatten(self.ds.dl_test)
        flat_X_train, flat_y_train = flat_train
        flat_X_test, flat_y_test = flat_test
        flat_X_train = flat_X_train.reshape(-1, 3 * 32 * 32)
        flat_X_test = flat_X_test.reshape(-1, 3 * 32 * 32)
        flat_X_train = pca.fit_transform(flat_X_train).reshape(5000, 1, 32, 32)
        flat_X_test = pca.transform(flat_X_test).reshape(1000, 1, 32, 32)
        train = data_utils.TensorDataset(torch.tensor(flat_X_train, dtype=torch.float32), flat_y_train)
        test = data_utils.TensorDataset(torch.tensor(flat_X_test, dtype=torch.float32), flat_y_test)
        self.ds.dl_train = data_utils.DataLoader(train, batch_size=100, shuffle=True)
        self.ds.dl_test = data_utils.DataLoader(test, batch_size=100, shuffle=True)
        self.set_custom_model(selected['lr'], selected['momentum'], selected['std'], input_size = 1024)
        train_res, test_res, train_time = self.train(num_epochs = 150, input_size = 1024)
        self.plot_fig('pca_whitening', 'pca_whitening', train_res, test_res)

    def set_dropout_and_weight_decay_experiment(self, dropout = 0.35, weight_decay = 2e-4):
        self.set_dropout_model(dropout=dropout, weight_decay=weight_decay)
        train_res, test_res, train_time = self.train(num_epochs = 150)
        self.plot_fig('dropout_0.35_decay_0.002', 'dropout_0.35_decay_0.002', train_res, test_res)

    def set_network_width_experiment(self, widths=(64, 1024, 4096)):
        train_results = []
        test_results = []
        name = 'network_width'
        for w in widths:
            name += '_' + str(w)
            self.set_custom_model(hidden_dim=w)
            train_res, test_res, _ = self.train(num_epochs = 150)
            train_results.append(train_res)
            test_results.append(test_res)
        self.comapre_plot_fig(name, 'width', widths, train_results, test_results)

    def set_network_depth_experiment(self, hidden_dim = 64, depths=(3, 4, 10)):
        train_results = []
        test_results = []
        name = 'network_depth'
        for d in depths:
            name += '_' + str(d)
            layers = [nn.Linear(self.in_features, hidden_dim)]
            for _ in range(d - 1):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, self.num_classes))
            self.set_custom_model(layers=layers)
            train_res, test_res, _ = self.train(num_epochs = 150)
            train_results.append(train_res)
            test_results.append(test_res)
        self.comapre_plot_fig(name, 'depth', depths, train_results, test_results)

    def train(self, num_epochs = 40, input_size = 3 * 32 * 32):
        # Train the Model
        self.model.to(self.device)
        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        test_res = Result(accuracy=[], loss=[])
        train_loader = self.ds.dl_train
        test_loader = self.ds.dl_test
        print("debug prints start: %d" % (num_epochs))
        start_train_loop = time.time()
        for epoch in range(num_epochs):
            print(epoch)
            avg_train_loss = avg_train_acc = avg_test_loss = avg_test_acc = 0.
            for i, (images, labels) in enumerate(train_loader):
                if self.device:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                images = Variable(images)
                labels = Variable(labels)
                # Forward + Backward + Optimize
                self.optimizer.zero_grad()
                preds = self.model(images)
                output_loss = self.criterion(preds, labels)
                output_loss.backward()
                self.optimizer.step()
                y_pred = torch.max(preds, dim=1).indices
                num_correct = torch.sum(labels == y_pred).item()
                avg_train_loss += output_loss.item() / len(train_loader)
                avg_train_acc += (num_correct / len(labels) / len(train_loader))
            train_res.loss.append(avg_train_loss)
            train_res.accuracy.append(avg_train_acc)

            for images, labels in test_loader:
                if self.device:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                with torch.no_grad():
                    images = Variable(images.view(-1, input_size))
                    labels = Variable(labels)
                    preds = self.model(images)
                    output_test_loss = self.criterion(preds, labels)
                    y_pred = torch.max(preds, dim=1).indices
                    num_correct = torch.sum(labels == y_pred).item()
                    avg_test_loss += output_test_loss.item() / len(test_loader)
                    avg_test_acc += (num_correct / len(labels) / len(test_loader))
            test_res.loss.append(avg_test_loss)
            test_res.accuracy.append(avg_test_acc)
        loop_time = time.time() - start_train_loop
        print('loop time: %d' % loop_time)
        print('Accuracy of the network on the train images: %d %%' % (100 * train_res.accuracy[-1]))
        print('Accuracy of the network on the test images: %d %%' % (100 * test_res.accuracy[-1]))
        return train_res, test_res, loop_time

    def evaluate(self, test_loader = None):
        total, correct = 0, 0
        if test_loader == None:
            test_loader = self.ds.dl_test
        for images, labels in test_loader:
            with torch.no_grad():
                images = Variable(images.view(-1, 3 * 32 * 32))
                total += labels.size(0)
                correct += torch.sum(self.model.forward(images).argmax(dim = 1) == labels)
                # Save the Model
                torch.save(self.model.state_dict(), 'config_1.pkl')
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    def plot_fig(self, filename, plot_title, train_res, test_res):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        fig.subplots_adjust(top=0.8)
        fig.suptitle(plot_title, fontsize=15, fontweight='bold', y=1)
        for i, loss_acc in enumerate(('loss', 'accuracy')):
            axes[i].plot(getattr(train_res, loss_acc))
            axes[i].plot(getattr(test_res, loss_acc))
            axes[i].set_title(loss_acc.capitalize(), fontweight='bold')
            axes[i].set_xlabel('Epoch')
            axes[i].legend(('train', 'test'))
            axes[i].grid(which='both', axis='y')

        plt.savefig(self.dir + '/part2_experiments/' + filename + '.png')
        plt.show()

    def comapre_plot_fig(self, filename, compared_param, param_values, train_results, test_results):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
        fig.subplots_adjust(top=0.8)
        fig.suptitle(filename, fontsize=15, fontweight='bold', y=1)
        leg = [str(compared_param) + '=' + str(s) for s in param_values]
        for i, loss_acc in enumerate(('loss', 'accuracy')):
            [axes[0][i].plot(getattr(train_res, loss_acc)) for train_res in train_results]
            axes[0][i].set_title('Train ' + loss_acc.capitalize(), fontweight='bold')
            axes[0][i].set_xlabel('Epoch')
            axes[0][i].legend((leg))
            axes[0][i].grid(which='both', axis='y')

        for i, loss_acc in enumerate(('loss', 'accuracy')):
            [axes[1][i].plot(getattr(test_res, loss_acc)) for test_res in test_results]
            axes[1][i].set_title('Test ' + loss_acc.capitalize(), fontweight='bold')
            axes[1][i].set_xlabel('Epoch')
            axes[1][i].legend(leg)
            axes[1][i].grid(which='both', axis='y')

        plt.savefig(self.dir + '/part2_experiments/' + filename + '.png')
        plt.show()


if __name__ == '__main__':
    p = Part_2(in_features, num_classes)
    train_res, test_res, train_time = p.train(num_epochs=150)



