# detector/apps.py
from django.apps import AppConfig
import tensorflow as tf 
import os
from django.conf import settings

# ==========================================
# THE Keras 3 Bug Fix (Monkey Patch)
# ==========================================
original_dense_init = tf.keras.layers.Dense.__init__

def custom_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None) 
    original_dense_init(self, *args, **kwargs)

tf.keras.layers.Dense.__init__ = custom_dense_init
# ==========================================

class DetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
    
    # Use dictionaries to store all the models!
    species_models = {}
    disease_models = {}

    def ready(self):
        base_dir = settings.BASE_DIR
        models_dir = os.path.join(base_dir, 'models')
        
        print("Loading 8 models into memory... this might take a few seconds...")
        
        # 🐟 Load Species Models
        DetectorConfig.species_models = {
            'ResNet50': tf.keras.models.load_model(os.path.join(models_dir, 'FishClass_ResNet50_99.h5')),
            'EfficientNetB0': tf.keras.models.load_model(os.path.join(models_dir, 'FishClasses_EfficientNetB0.h5')),
            'DenseNet121': tf.keras.models.load_model(os.path.join(models_dir, 'FishClasses_DenseNet121.h5')),
            'Custom': tf.keras.models.load_model(os.path.join(models_dir, 'FishClasses_Custom.h5')),
        }
        
        # 🦠 Load Disease Models
        DetectorConfig.disease_models = {
            'ResNet50': tf.keras.models.load_model(os.path.join(models_dir, 'FishDisease_ResNet50.h5')),
            'EfficientNetB0': tf.keras.models.load_model(os.path.join(models_dir, 'fish_disease_EfficientNetB0_finetuned.h5')),
            'DenseNet121': tf.keras.models.load_model(os.path.join(models_dir, 'FishDisease_DenseNet121.h5')),
            'Custom': tf.keras.models.load_model(os.path.join(models_dir, 'FishDisease_Custom.h5')),
        }
        
        print("All 8 Models loaded successfully!")