import matplotlib.pyplot as plt
from model.model import Model
from input.download_data import download_mnist_datasets

parameters_path = "model/parameters.json"
train_set, test_set = download_mnist_datasets()

M = Model(parameters_path=parameters_path)
M.do_train(train_set, test_set)
M.save(inplace=True)
M.history.plot()
plt.show()
