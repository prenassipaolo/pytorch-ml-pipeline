import importlib
import json

class Loss:
    """
    A class used to load a loss function

    ...
    Attributes
    ----------
    LOSS_NAME : str
        Name of the loss class
    LOSS_PATH : str
        Path of the file containing the loss class
    PARAMETERS_PATH: str
        Path of the json file containing the parameters

    Methods
    -------
    get_loss()
        loads LOSS_NAME class according to LOSS_PATH
    get_parameters()
        loads class parameters according to PARAMETERS_PATH
    def create_function()
        creates an instance of the function class with the defined parameters
    """
    def __init__(self, LOSS_NAME, LOSS_PATH, PARAMETERS_PATH):
        """
        Parameters
        ----------
        LOSS_NAME : str
            Name of the loss class
        LOSS_PATH : str
            Path of the file containing the loss class
        PARAMETERS_PATH: str
            Path of the json file containing the parameters
        """
        
        self.loss_name = LOSS_NAME
        self.loss_path = LOSS_PATH
        self.parameters_path = PARAMETERS_PATH
        self.create_function()

    def get_loss(self):
        """
        Loads LOSS_NAME class according to LOSS_PATH
        
        Returns
        ----------
        class:
            Instance of the loss class according to the declared paths
        """
        loss_path_aux = self.loss_path
        if __name__=="__main__":
            loss_path_aux = "/".join(self.loss_path.split('/')[1:])
        aux = loss_path_aux.split('/')
        aux[-1] = aux[-1].split(".")[0]
        aux = ".".join(aux)
        loss = importlib.import_module(aux)
        loss_class = getattr(loss, self.loss_name)
        return loss_class

    def get_parameters(self):
        """
        Loads class parameters according to PARAMETERS_PATH

        Returns
        ----------
        dict:
            Parameters of the loss function
        """
        
        parameters_path_aux = self.parameters_path
        
        #if __name__=="__main__":
        #    parameters_path_aux = "/".join(self.parameters_path.split('/')[1:])
        
        with open(parameters_path_aux, 'r') as f:
            p = json.load(f)
        
        return p

    def create_function(self):
        """
        Creates an instance of the function class with the defined parameters
        """

        self.parameters = self.get_parameters()
        self.function = self.get_loss()(self.parameters)
        
        return

    def __call__(self):
        """Loads the loss function given the path of file containing the loss class and 
        the name of the loss class
        
        Returns
        ----------
        function:
            Loss function
        """

        return self.function

"""
### EXAMPLE

LOSS_NAME="BinaryCrossEntropy"
LOSS_PATH="model/losses/binarycrossentropy.py"
PARAMETERS_PATH = "model/losses/binarycrossentropy_parameters.json"

a = Loss(LOSS_NAME, LOSS_PATH, PARAMETERS_PATH)

print(1, a)
print(2, a())
"""
