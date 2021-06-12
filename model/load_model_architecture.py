import importlib


def load_model_architecture(MODEL_NAME, MODEL_PATH):
    """Loads the model architecture class given the path of file containing the model architecture and 
    the name of the model class

    Parameters
    ----------
    MODEL_NAME : str
        Name of the model class
    MODEL_PATH : str
        Path of the file containing the model architecture
    
    Returns
    ----------
    class: torch.nn.Module
        Model architecture class


    """

    if __name__=="__main__":
        MODEL_PATH = "/".join(MODEL_PATH.split('/')[1:])
    
    aux = MODEL_PATH.split('/')
    aux[-1] = aux[-1].split(".")[0]
    aux = ".".join(aux)
    module = importlib.import_module(aux)
    model_class = getattr(module, MODEL_NAME)

    return model_class


### EXAMPLE
"""
MODEL_NAME="FeedForwardNet"
MODEL_PATH="model/architectures/feedforwardnet.py"

model_class = load_model_architecture(MODEL_NAME=MODEL_NAME, MODEL_PATH=MODEL_PATH)
print(model_class)
"""
