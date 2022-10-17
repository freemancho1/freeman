from sys import displayhook
import tensorflow as tf
from utils.support_tf import LogLevelManager as llm


def check_gpus(log_level=3, is_display_message=True, is_memory_growth=True):
    llm.set(log_level)
    
    p_gpus, l_gpus = [], []
    display_msg = ""
    p_gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if p_gpus:
        try: 
            for gpu in p_gpus:
                tf.config.experimental.set_memory_growth(gpu, is_memory_growth)
            l_gpus = tf.config.experimental.list_logical_devices('GPU')
            display_msg = f"{len(p_gpus)} Physical Devices, {len(l_gpus)} Logical Devices"
        except RuntimeError as e:
            display_msg = str(e)
    else:
        display_msg = "No GPU device found"
        
    if is_display_message:
        print(display_msg)
        
    llm.reset()
    return (p_gpus, l_gpus)
