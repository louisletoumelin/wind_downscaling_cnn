from Models import VCD
from Models import UNet

'''
Modify Prm
'''


def create_prm_model(prm):
    """
    Select the desired model

    Input: str, ex: 'UNet'
    Output: function, ex UNet.Unet()
    """
    if prm['model'] == 'UNet':
        prm['model_func'] = UNet.Unet(prm)
    if prm['model'] == 'VCD':
        prm['model_func'] = VCD.VCD(prm)
    return(prm)
