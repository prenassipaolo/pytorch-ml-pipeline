from model.model import Model
from data.download_data import download_mnist_datasets

parameters_path = "model/default_parameters.json"
train_set, test_set = download_mnist_datasets()

M = Model(parameters_path=parameters_path)
M.do_train(train_set, test_set)
print(M.history)
M.save(inplace=True)
