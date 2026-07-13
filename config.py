import torch

class Config ():

    IMAGE_DIR = r"C:\Users\E-PART.iR\Desktop\templates"
    SAVE_MODEL_PATH = r"C:\Users\E-PART.iR\Desktop\new_prejects\Best_Modelresnet50.pth"
    EPOCHS = 20
    INPUT_SIZE = 224
    NUM_CLASS = 8
    LEARNING_RATE = 0.0001
    PATIENCE = 5
    BATCH_SIZE = 16
    DIVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
