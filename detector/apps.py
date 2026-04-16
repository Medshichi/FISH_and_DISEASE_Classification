# detector/apps.py
from django.apps import AppConfig
import tensorflow as tf 
import os
from django.conf import settings


original_dense_init = tf.keras.layers.Dense.__init__

def custom_dense_init(self, *args, **kwargs):
    # If the buggy Keras 3 tag is in the file, delete it
    kwargs.pop('quantization_config', None) 
    original_dense_init(self, *args, **kwargs)

# Overwrite the default layer with our custom one
tf.keras.layers.Dense.__init__ = custom_dense_init


class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
    
    type_model = None
    disease_model = None

    def ready(self):
        base_dir = settings.BASE_DIR
        
        # Load the models. The patch above will protect them from crashing!
        DetectorConfig.type_model = tf.keras.models.load_model(os.path.join(base_dir, 'models', 'FishClass_ResNet50_99.h5'))
        DetectorConfig.disease_model = tf.keras.models.load_model(os.path.join(base_dir, 'models', 'fish_disease_EfficientNetB0_finetuned.h5'))
        
        print("Models loaded successfully!")