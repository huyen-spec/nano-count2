# importing the sys module
import sys        
 
# appending the directory of mod.py
# in the sys.path list
sys.path.append('/home/huyentn2/huyen/project/project_nano_count/segmentation_unet/')
from dataset import BasicDataset, PramDataset      

# obj = BasicDataset()

from training import train_model
from evaluate import inference