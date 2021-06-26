import importlib
import json
import os
import pickle
import pandas as pd
import torch

class Model:
    def __init__(self, parameters=None, parameters_path: str=None):

        # parameters
        self.parameters_path = parameters_path
        self.parameters = parameters
        self.update_parameters()
        # items
        self.architecture = self.create_item_instance("architecture")
        self.criterion = self.create_item_instance("criterion")
        self.optimizer = self.create_item_instance("optimizer")
        self.scheduler = self.create_item_instance("scheduler")
        self.train = self.create_item_instance("train")
        self.earlystopping = self.create_item_instance("earlystopping")
        # outputs
        self.history = pd.DataFrame()

    def update_parameters(self):
        if self.parameters_path:
            with open(self.parameters_path, 'r') as f:
                self.parameters = json.load(f)
        return None

    def get_item_class(self, item):
        aux = self.parameters[item]["PATH"].split('/')
        aux[-1] = aux[-1].split(".")[0]
        aux = ".".join(aux)
        module = importlib.import_module(aux)
        item_class = getattr(module, self.parameters[item]["NAME"])
        return item_class
        
    def create_item_instance(self, item):
        if self.parameters:
            if item in self.parameters.keys():
                item_class = self.get_item_class(item)
                return item_class(**self.parameters[item]["PARAMETERS"])
        return None

    def save(self, filename='model', path_folder='./output/', pickle_file=False,\
         json_file=True, weights_file=True, history_file=True, inplace=False):

        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
            print("Directory " , path_folder ,  " created ")

        # check if the user wants to overwrite the file in case it exists
        if inplace:
            filepath = path_folder + f'{filename}' + '.{ext}'
        else:
            # find file "model_N.EXT" with maximum N
            aux = [f for f in os.listdir(path_folder) if os.path.isfile(''.join([path_folder, f]))]
            aux = [f for f in aux if f[:len(filename)]==f'{filename}']
            aux = [f[len(filename):].split('.')[0] for f in aux]
            aux = [f.strip('_').split('_')[0] for f in aux]
            aux_numeric =  [int(f) for f in aux if f.isnumeric()]
            if aux_numeric:
                filepath = path_folder + f'{filename}_{max(aux_numeric) + 1}' + '.{ext}'
                print('empty', filepath)
            elif '' in aux:
                filepath = path_folder + f'{filename}_{1}' + '.{ext}'
                print('empty', filepath)
            else:
                filepath = path_folder + f'{filename}' + '.{ext}'

        # save parameters into .json file
        if json_file:
            # check if there are parameters to store
            if self.parameters:
                with open(filepath.format(ext='json'), 'w') as fp:
                    json.dump(self.parameters, fp)
            else:
                print('No parameters to store. Fill the parameters \
                    attribute before saving into json file.')
        # save model weights into .h5 file
        if weights_file:
            torch.save(
                self.architecture.state_dict(), 
                filepath.format(ext='h5')
                )
        # save model into .p file
        if pickle_file:
            with open(filepath.format(ext='p'), 'wb') as fp:
                pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # save history into .csv file
        if history_file:
            # check if there are values to store
            if not self.history.empty:
                self.history.to_csv(filepath.format(ext='csv'))
            else:
                print('No history to store. Train the model \
                    before saving into csv file.')
        return

    def do_train(self, train_set, test_set):
        self.history = self.train(self, train_set, test_set)
        return self.history


### EXAMPLES
# model creation
'''
parameters_path = "model/default_parameters.json"
M = Model(parameters_path=parameters_path)
print(M.train.__dict__)
print(M.architecture.__dict__)
'''

# model items
'''
print("---Model\n", M)
print("---architecture\n", M.architecture)
print("---criterion\n", M.criterion)
print("---optimizer\n", M.optimizer)
print("---scheduler\n", M.scheduler)
'''

# model without parameters
'''
N = Model()
N.architecture = M.architecture
N.criterion = M.criterion
N.optimizer = M.optimizer
N.scheduler = M.scheduler
'''

# what can be stored in a model
'''
print("---__class__\n", N.__class__)
print("---__dict__\n", N.__dict__)
print("---architecture.__class__\n", N.architecture.__class__)
print("---architecture.__dict__\n", N.architecture.__dict__)
print("---weights\n", N.architecture.state_dict())
'''

# empty model
'''
N = Model()
print("---Model\n", N)
print("---architecture\n", N.architecture)
print("---criterion\n", N.criterion)
print("---optimizer\n", N.optimizer)
print("---scheduler\n", N.scheduler)
'''

# save
'''
M.save(filename='')
'''