import torch
from helpers.models.InvNet import InvNet


class Model:
        
    def __init__(self, model_path) :
        '''
        Args:
            model_path (List[str]): the path to the model file (.pth, .pt, \
                or .h5) containing the trained parameters. 
            type (str)(optional)(default='combined'): the system currently will accept either a \
                combined model or separated model for predicting valence and \
                arousal. Preparing to accept classification in the future.
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.type = type
        if (self.device=='cpu'):
            ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        else :
            ckpt = torch.load(model_path)
        self.model = InvNet().to(self.device) # init model accordingly here
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

    def predict(self, input) :
        '''
        Args:
            input (torch.Tensor): tensor of shape NxCxHxW \
                representing the cropped-face images from \
                the users
                - N: the number of faces in the original photo
                - C: the number of channels (3)
                - H = W: the height and width of the cropped \
                    face image. Resized to 224x244 using function \
                    in face_detection.py
        Return:
            torch.Tensor: tensor of shape Nx1X2 including the \
                valence and arousal values of each input images.
        '''
        input = input.to(self.device)
        with torch.no_grad() :
            output = self.model.forward(input)
            return output
    

    