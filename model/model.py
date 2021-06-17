import importlib
import json



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



""""""
### EXAMPLE

model_parameters_path = "model/model_parameters.json"

M = Model(model_parameters_path=model_parameters_path)
N = Model()

print("---Model\n", M)
print("---architecture\n", M.architecture)
print("---loss\n", M.loss)
print("---optimizer\n", M.optimizer)
print("---scheduler\n", M.scheduler)

print("---Model\n", N)
print("---architecture\n", N.architecture)
print("---loss\n", N.loss)
print("---optimizer\n", N.optimizer)
print("---scheduler\n", N.scheduler)

