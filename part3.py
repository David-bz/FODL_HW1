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


in_features= 3 * 32 * 32
num_classes = 10
selected = {'lr' : 0.0075,
            'momentum' : 0.8,
            'std' : 0.05}


class CNN_Wrapper(nn.Module):
    def __init__(self, in_size, num_classes, channels, hidden_dims, init_std = selected['std'], dropout=None, num_conv_layers = 2):
        def init_weights(layer):
            if type(layer) in [nn.Linear, nn.Conv2d]:
                torch.nn.init.normal_(layer.weight, mean=0, std = init_std)
                layer.bias.data.fill_(0.01)
        super().__init__()
        self.in_channels = in_size
        self.num_classes = num_classes

        self.channels = channels
        self.hidden_dims = hidden_dims
        feature_layers = [nn.Conv2d(self.in_channels, channels[0], kernel_size=3, stride=1),
                  nn.ReLU(),
                  torch.nn.MaxPool2d(2, stride=2)]
        for _ in range(num_conv_layers - 2):
            feature_layers += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                  nn.ReLU()]
        feature_layers += [
                  nn.Conv2d(channels[-2], channels[-1], kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                  torch.nn.MaxPool2d(2, stride=2)]
        if dropout != None:
            conv_indices = [i for i,mod in enumerate(feature_layers) if type(mod) == nn.Conv2d]
            for c in conv_indices:
                feature_layers.insert(c + 1, nn.Dropout2d(dropout))
            print(feature_layers)
        self.feature_models = nn.Sequential(*feature_layers)
        classifier_layers = [nn.Linear(hidden_dims[0], hidden_dims[1]),
                  nn.Linear(hidden_dims[1], self.num_classes)]
        self.classifier = nn.Sequential(*classifier_layers)
        self.feature_models.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, X):
        features = self.feature_models(X)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out

class Part_3:
    def __init__(self, in_features, num_classes, init_model=True):
        torch.manual_seed(10)
        self.ds = HW1_Dataset(batch_size = 100, subset_portion = 0.1)
        self.in_features, self.num_classes =  in_features, num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dir = os.path.abspath('./')
        if init_model:
            self.set_baseline_model()

    def set_baseline_model(self, input_size = (3, 32, 32), channels = (64, 16), hidden_dims = (784,784),
                            init_lr = selected['lr'], init_momentum = selected['lr'],
                           init_std = selected['std'], dropout=None, init_weight_decay=0., num_conv_layers = 2):
        num_classes = 10
        in_channels, in_h, in_w = input_size

        self.model = CNN_Wrapper(in_channels, num_classes, channels, hidden_dims, init_std, dropout, num_conv_layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=init_lr, momentum=init_momentum, weight_decay=init_weight_decay)

    def perform_grid_search(self, subset_portion = 0.1, num_epochs=100):
        self.ds = HW1_Dataset(batch_size = 100, subset_portion = subset_portion)
        lr_values = [1e-2]
        momentum_values = [0.9]
        std_values = [0.02]
        res = []
        for lr in lr_values:
            for momentum in momentum_values:
                for std in std_values:
                    print(lr, momentum, std)
                    self.set_baseline_model(init_lr=lr, init_momentum=momentum, init_std=std)
                    train_res, test_res, training_time = self.train(num_epochs=num_epochs)
                    name = 'lr_{}_momentum_{}_std_{}'.format(lr, momentum, std)
                    self.plot_fig(name, name, train_res, test_res)
                    res.append([lr, momentum, std] + np.mean(train_res.accuracy[-3:]) + np.mean(train_res.loss[-3:]) +
                               np.mean(test_res.accuracy[-3:]) + np.mean(test_res.loss[-3:]) +
                               [training_time])
        with open("part3_grid_res.pkl", "wb+") as f:
            pickle.dump(res, f)

    def set_optimization_experiment(self, lr = 0.0006):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        train_res, test_res, train_time = self.train(num_epochs = 100)
        self.plot_fig('adam_optimizer_' + str(lr), 'adam_optimizer_' + str(lr), train_res, test_res)

    def set_initialization_experiment(self):
        def apply_xavier(layer):
            if type(layer) in [nn.Linear, nn.Conv2d]:
                nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(0.01)

        self.model.feature_models.apply(apply_xavier)
        self.model.classifier.apply(apply_xavier)
        self.optimizer = optim.SGD(self.model.parameters(), lr=selected['lr'], momentum=selected['momentum'])
        train_res, test_res, train_time = self.train(num_epochs = 100)
        self.plot_fig('xavier_init_', 'xavier_init_', train_res, test_res)

    def set_dropout_and_weight_decay_experiment(self, dropout = 0.15, weight_decay = 1e-4):
        self.set_baseline_model(dropout=dropout, init_weight_decay=weight_decay)
        train_res, test_res, train_time = self.train(num_epochs = 100)
        self.plot_fig('dropout_' + str(dropout) + '_decay_' + str(weight_decay),
                      'dropout_' + str(dropout) + '_decay_' + str(weight_decay), train_res, test_res)

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
        self.set_baseline_model(input_size=(1, 32, 32))
        train_res, test_res, train_time = self.train(num_epochs = 100)
        self.plot_fig('pca_preprocess_' + str(n_comp), 'pca_preprocess_' + str(n_comp), train_res, test_res)

    def set_network_width_experiment(self, filter_sizes=((256, 64), (512, 256))):
        train_results = []
        test_results = []
        name = 'network_width'
        linear_dimension = [3136, 12544]
        for channels, dim in zip(filter_sizes, linear_dimension):
            name += '_' + str(channels)
            self.set_baseline_model(channels=channels, hidden_dims=(dim, 784))
            train_res, test_res, _ = self.train(num_epochs = 100)
            train_results.append(train_res)
            test_results.append(test_res)
        self.comapre_plot_fig(name, 'width', filter_sizes, train_results, test_results)

    def set_network_depth_experiment(self, max_k = 5):
        train_results = []
        test_results = []
        name = 'network_depth'
        linear_dimension = [3136, 12544]
        for k in range(2, max_k + 1):
            name += '_' + str(k)
            self.set_baseline_model(num_conv_layers=k)
            train_res, test_res, _ = self.train(num_epochs = 100)
            train_results.append(train_res)
            test_results.append(test_res)
        self.comapre_plot_fig(name, 'depth', list(range(2, max_k + 1)), train_results, test_results)

    def train(self, num_epochs = 100):
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

        plt.savefig(self.dir + 'part3_experiments/' + filename + '.png')
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

        plt.savefig(self.dir + '/experiments/' + filename + '.png')
        plt.show()

    def run_hw4_experiment(self, filter_sizes=((512, 256),)):
        train_results = []
        test_results = []
        name = '/network_width/'
        linear_dimension = [12544]
        print(filter_sizes)
        print(type(filter_sizes))
        for channels, dim in zip(filter_sizes, linear_dimension):
            name += '_' + str(channels)
            self.set_baseline_model(channels=channels, hidden_dims=(dim, 784), num_conv_layers = 2)
            train_res, test_res, _ = self.train(num_epochs = 100)
            train_results.append(train_res)
            test_results.append(test_res)
        self.comapre_plot_fig(name, 'width', filter_sizes, train_results, test_results)


if __name__ == '__main__':
    p = Part_3(in_features, num_classes)
    # f_exe =[p.perform_grid_search,
    #         p.set_optimization_experiment,
    #         p.set_initialization_experiment,
    #         p.set_dropout_and_weight_decay_experiment,
    #         p.set_pca_experiment,
    #         p.set_network_width_experiment,
    #         p.set_network_depth_experiment]
    # for f in f_exe:
    #     p = Part_3(in_features, num_classes)
    #     f()
    p.run_hw4_experiment()
