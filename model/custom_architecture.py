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
    PARAMETERS_PATH:
        Path of the json file containing the parameters

    Methods
    -------
    get_class()
        loads MODEL_NAME architecture according to MODEL_PATH
    get_parameters()
        loads architecture parameters according to PARAMETERS_PATH
    def create_architecture()
        creates an instance of the architecture class, with the defined parameters
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
        """
        Loads MODEL_NAME class according to MODEL_PATH

        Returns
        ----------
        class:
            Instance of the architecture class according to the declared path
        """

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
        """
        Loads class parameters according to PARAMETERS_PATH

        Returns
        ----------
        dict:
            Parameters of the architecture
        """

        parameters_path_aux = self.parameters_path
        #if __name__=="__main__":
        #    parameters_path_aux = "/".join(self.parameters_path.split('/')[1:])
        with open(parameters_path_aux, 'r') as f:
            p = json.load(f)
        return p

    def create_architecture(self):
        """
        Creates an instance of the architecture class with the defined parameters
        """
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

"""
### EXAMPLE

MODEL_NAME="FeedForwardNet"
MODEL_PATH="model/architectures/feedforwardnet.py"
PARAMETERS_PATH = "model/architectures/feedforwardnet_parameters.json"
#model_class = build_model(MODEL_NAME=MODEL_NAME, MODEL_PATH=MODEL_PATH)
#print(model_class)

a = Architecture(MODEL_NAME, MODEL_PATH, PARAMETERS_PATH)

print(1, a)
print(2, a())
"""