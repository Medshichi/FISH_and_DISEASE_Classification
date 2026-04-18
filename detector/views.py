# detector/views.py
from django.shortcuts import render
from .apps import DetectorConfig
from PIL import Image
import numpy as np
import base64
import io

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess


FISH_CLASSES = {
    0: 'MilkFish',
    1: 'Tilapia'
}

DISEASE_CLASSES = {
    0: 'Bacterial Red disease',
    1: 'Bacterial gill disease',
    2: 'Fungal diseases Saprolegniasis',
    3: 'Healthy'
}


def custom_preprocess(img_array):
    return img_array / 255.0


PREPROCESSORS = {
    'ResNet50': resnet_preprocess,
    'EfficientNetB0': effnet_preprocess,
    'DenseNet121': densenet_preprocess,
    'Custom': custom_preprocess
}


def home_view(request):
    context = {
        'used_species_model': 'ResNet50',
        'used_disease_model': 'EfficientNetB0'
    }

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            img = Image.open(image_file).convert('RGB')

            # Preserve uploaded image preview
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            context['uploaded_image'] = f"data:image/jpeg;base64,{img_b64}"

            # User selected models
            selected_species_model = request.POST.get('species_model', 'ResNet50')
            selected_disease_model = request.POST.get('disease_model', 'EfficientNetB0')

            context['used_species_model'] = selected_species_model
            context['used_disease_model'] = selected_disease_model

            # Validate models
            species_preprocessor = PREPROCESSORS.get(selected_species_model)
            disease_preprocessor = PREPROCESSORS.get(selected_disease_model)

            species_model = DetectorConfig.species_models.get(selected_species_model)
            disease_model = DetectorConfig.disease_models.get(selected_disease_model)

            if not species_preprocessor or not disease_preprocessor:
                raise ValueError("Invalid preprocessing selected.")

            if not species_model or not disease_model:
                raise ValueError("Invalid AI model selected.")

            # Resize image
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)

            species_input = np.expand_dims(img_array.copy(), axis=0)
            disease_input = np.expand_dims(img_array.copy(), axis=0)

            # Apply proper preprocessing
            species_input = species_preprocessor(species_input)
            disease_input = disease_preprocessor(disease_input)

            # Debug test
            print("Selected species model:", selected_species_model)
            print("Selected disease model:", selected_disease_model)

            # Predictions
            type_pred = species_model.predict(species_input)
            disease_pred = disease_model.predict(disease_input)

            type_idx = np.argmax(type_pred[0])
            disease_idx = np.argmax(disease_pred[0])

            context['result'] = {
                'fish': FISH_CLASSES.get(type_idx, "Unknown"),
                'disease': DISEASE_CLASSES.get(disease_idx, "Unknown"),
                'conf_fish': round(float(np.max(type_pred[0])) * 100, 2),
                'conf_disease': round(float(np.max(disease_pred[0])) * 100, 2),
            }

        except Exception as e:
            context['error'] = f"Error processing image: {str(e)}"

    return render(request, 'detector/index.html', context)


TREATMENT_PLANS = {
    'Bacterial Red disease': {
        'title': 'Treating Bacterial Red Disease (Pond Scale)',
        'description': 'Typically caused by Aeromonas bacteria.',
        'steps': [
            'Halt feeding immediately.',
            'Perform partial water exchange.',
            'Apply agricultural lime.',
            'Use medicated feeds if necessary.'
        ]
    },
    'Bacterial gill disease': {
        'title': 'Treating Bacterial Gill Disease',
        'description': 'Caused by poor water conditions.',
        'steps': [
            'Increase oxygen immediately.',
            'Reduce feeding.',
            'Apply potassium permanganate carefully.',
            'Reduce overcrowding.'
        ]
    },
    'Fungal diseases Saprolegniasis': {
        'title': 'Treating Saprolegniasis',
        'description': 'Secondary fungal infection.',
        'steps': [
            'Avoid unnecessary handling.',
            'Remove infected fish.',
            'Maintain proper pond depth.',
            'Improve water quality.'
        ]
    }
}


def treatment_view(request, disease_name):
    plan = TREATMENT_PLANS.get(disease_name)

    return render(request, 'detector/treatment.html', {
        'disease': disease_name,
        'plan': plan
    })