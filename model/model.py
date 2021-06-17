import importlib
import json
import pickle
import os
import torch



class Model:
    def __init__(self, model_parameters=None, model_parameters_path: str=None):

        # parameters
        self.model_parameters_path = model_parameters_path
        self.model_parameters = model_parameters
        self.update_model_parameters()
        # items
        self.architecture = self.create_item_instance("architecture")
        self.loss = self.create_item_instance("loss")
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
    


    def update_model_parameters(self):
        if self.model_parameters_path:
            with open(self.model_parameters_path, 'r') as f:
                self.model_parameters = json.load(f)
        return
    
    def get_item_class(self, item):
        aux = self.model_parameters[item]["PATH"].split('/')
        aux[-1] = aux[-1].split(".")[0]
        aux = ".".join(aux)
        module = importlib.import_module(aux)
        item_class = getattr(module, self.model_parameters[item]["NAME"])
        return item_class
    
    def create_item_instance(self, item):
        if self.model_parameters:
            if item in self.model_parameters.keys():
                item_class = self.get_item_class(item)
                return item_class(self.model_parameters[item]["PARAMETERS"])
        return
    
    def create_optimizer(self):
        item_class = self.create_item_instance('optimizer')
        if item_class and self.architecture:
            return item_class(self.architecture.parameters())
        return 
    
    def create_scheduler(self):
        item_class = self.create_item_instance('scheduler')
        if item_class and self.optimizer:
            return item_class(self.optimizer)
        return 
    
    def save_weights(self, path=None):
        saved = False
        if path:
            torch.save(
                self.architecture.state_dict(), 
                path
                )
            saved = True
        elif self.model_parameters:
            if 'save' in self.model_parameters.keys():
                if 'PATH_WEIGHTS' in self.model_parameters['save'].keys():
                    torch.save(
                        self.architecture.state_dict(), 
                        self.model_parameters['save']['PATH_WEIGHTS']
                        )
                    saved = True
        if not saved:
            print('\nFile not saved. No path provided\n')
        return saved

    def save_pickle(self, path=None):
        
        saved = False
        
        if path:
            with open(path, 'wb') as fp:
                pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)
            saved = True
        elif self.model_parameters:
            if 'save' in self.model_parameters.keys():
                if 'PATH_PICKLE' in self.model_parameters['save'].keys():
                    with open(self.model_parameters['save']['PATH_PICKLE'], 'wb') as fp:
                        pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    saved = True
        if not saved:
            print('\nFile not saved. No path provided.\n')
        
        return saved
    
    def save(self, filename='model', path_folder='./outputs/', pickle_file=False, json_file=True, weights_file=True, inplace=False):
        
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
            print(aux)
            aux = [f.strip('_').split('_')[0] for f in aux]
            print(aux)
            aux_numeric =  [int(f) for f in aux if f.isnumeric()]
            if aux_numeric:
                filepath = path_folder + f'{filename}_{max(aux_numeric) + 1}' + '.{ext}'
                print('empty', filepath)
            elif '' in aux:
                filepath = path_folder + f'{filename}_{1}' + '.{ext}'
                print('empty', filepath)
            else:
                filepath = path_folder + f'{filename}' + '.{ext}'

        # save model_parameters into .json file
        if json_file:
            # check if there are parameters to store
            if self.model_parameters:
                with open(filepath.format(ext='json'), 'w') as fp:
                    json.dump(self.model_parameters, fp)
            else:    
                print('No parameters to store. Fill the model_parameters attribute before saving into json file.')
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
        
        return


        
        
            


### EXAMPLES
# model creation

"""
model_parameters_path = "model/model_parameters.json"

M = Model(model_parameters_path=model_parameters_path)
"""

# model items
'''
print("---Model\n", M)
print("---architecture\n", M.architecture)
print("---loss\n", M.loss)
print("---optimizer\n", M.optimizer)
print("---scheduler\n", M.scheduler)
'''

# model without parameters
'''
N = Model()

N.architecture = M.architecture
N.loss = M.loss
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
"""
N = Model()

print("---Model\n", N)
print("---architecture\n", N.architecture)
print("---loss\n", N.loss)
print("---optimizer\n", N.optimizer)
print("---scheduler\n", N.scheduler)
"""
# save
"""
M.save(filename='')
"""
