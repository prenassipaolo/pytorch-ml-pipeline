import importlib
import json

class Architecture:
    """
    A class used to load an architecture

    ...
    Attributes
    ----------
    MODEL_NAME : str
            Name of the model class
    MODEL_PATH : str
            Path of the file containing the model architecture
    PARAMETERS_PATH

    Methods
    -------
    get_class()
        load MODEL_NAME architecture according to MODEL_PATH
    get_parameters()
        load architecture parameters according to PARAMETERS_PATH
    def create_architecture()
        create an instance of the architecture class, with the defined parameters
    """
    def __init__(self, MODEL_NAME, MODEL_PATH, PARAMETERS_PATH):
        """
        Parameters
        ----------
        MODEL_NAME : str
            Name of the model class
        MODEL_PATH : str
            Path of the file containing the model architecture
        """
        
        self.model_name = MODEL_NAME
        self.model_path = MODEL_PATH
        self.parameters_path = PARAMETERS_PATH
        self.create_architecture()

    def get_class(self):
        model_path_aux = self.model_path
        if __name__=="__main__":
            model_path_aux = "/".join(self.model_path.split('/')[1:])
        aux = model_path_aux.split('/')
        aux[-1] = aux[-1].split(".")[0]
        aux = ".".join(aux)
        module = importlib.import_module(aux)
        model_class = getattr(module, self.model_name)
        return model_class

    def get_parameters(self):
        parameters_path_aux = self.parameters_path
        #if __name__=="__main__":
        #    parameters_path_aux = "/".join(self.parameters_path.split('/')[1:])
        with open(parameters_path_aux, 'r') as f:
            p = json.load(f)
        return p

    def create_architecture(self):
        self.parameters = self.get_parameters()
        self.architecture = self.get_class()(self.parameters)
        return

    def __call__(self):
        """Loads the model architecture class given the path of file containing the model architecture and 
        the name of the model class
        
        Returns
        ----------
        class: torch.nn.Module
            Model architecture class
        """

        return self.architecture




### EXAMPLE

MODEL_NAME="FeedForwardNet"
MODEL_PATH="model/architectures/feedforwardnet.py"
PARAMETERS_PATH = "model/architectures/feedforwardnet_parameters.json"
#model_class = build_model(MODEL_NAME=MODEL_NAME, MODEL_PATH=MODEL_PATH)
#print(model_class)

a = Architecture(MODEL_NAME, MODEL_PATH, PARAMETERS_PATH)
print(a.architecture)