# detector/views.py
from django.shortcuts import render
from .apps import DetectorConfig
from PIL import Image
import numpy as np
import base64
import io

# Import ALL the specific Keras Preprocessors
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

FISH_CLASSES = {0: 'MilkFish', 1: 'Tilapia'} 
DISEASE_CLASSES = {0: 'Bacterial Red disease', 1: 'Bacterial gill disease', 2: 'Fungal diseases Saprolegniasis', 3: 'Healthy'}

# Custom preprocessing for your CNN trained from scratch
def custom_preprocess(img_array):
    return img_array / 255.0

# Map the dropdown names to their exact math formulas!
PREPROCESSORS = {
    'ResNet50': resnet_preprocess,
    'EfficientNetB0': effnet_preprocess,
    'DenseNet121': densenet_preprocess,
    'Custom': custom_preprocess
}

def home_view(request):
    context = {}

    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            img = Image.open(image_file).convert('RGB')
            
            # Base64 Image Preview
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            context['uploaded_image'] = f"data:image/jpeg;base64,{img_b64}"

            # Grab what the user selected from the dropdowns (Default to ResNet/EfficientNet)
            selected_species_model = request.POST.get('species_model', 'ResNet50')
            selected_disease_model = request.POST.get('disease_model', 'EfficientNetB0')

            # Pass the selections back to the HTML so we know what was used
            context['used_species_model'] = selected_species_model
            context['used_disease_model'] = selected_disease_model

            # Resize
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) 
            
            species_input = np.expand_dims(img_array.copy(), axis=0)
            disease_input = np.expand_dims(img_array.copy(), axis=0)

            # Apply the EXACT math needed for the model they chose
            species_input = PREPROCESSORS[selected_species_model](species_input)
            disease_input = PREPROCESSORS[selected_disease_model](disease_input)

            # Predict using the chosen models!
            type_pred = DetectorConfig.species_models[selected_species_model].predict(species_input)
            disease_pred = DetectorConfig.disease_models[selected_disease_model].predict(disease_input)

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

# Here we prepare the step on how auqa culture farms owner can mitigate the detected disease in the fish
TREATMENT_PLANS = {
    'Bacterial Red disease': {
        'title': 'Treating Bacterial Red Disease (Pond Scale)',
        'description': 'Typically caused by Aeromonas bacteria. In a pond setting, this spreads rapidly when the bottom soil degrades and water quality drops.',
        'steps': [
            'Halt feeding immediately to prevent further water quality degradation.',
            'Perform a partial water exchange by flushing the pond with fresh, clean water for several days.',
            'Broadcast agricultural lime (agrilime) across the pond to stabilize the pH and sanitize the bottom sludge.',
            'Once water quality stabilizes, administer medicated floating feeds containing permitted antibiotics (consult a local marine vet for commercial dosing).'
        ]
    },
    'Bacterial gill disease': {
        'title': 'Treating Bacterial Gill Disease (Pond Scale)',
        'description': 'A highly contagious infection that attacks the gills, usually triggered by overcrowded ponds and heavy organic buildup.',
        'steps': [
            'Immediately turn on all paddlewheel aerators to maximize dissolved oxygen, especially at night when oxygen levels crash.',
            'Stop pond fertilization (manure/chemical) and feeding until mortality rates drop.',
            'Carefully broadcast Potassium Permanganate (KMnO4) at 2-4 ppm across the pond to reduce the bacterial load in the water column.',
            'If the pond is overcrowded, urgently reduce the stocking density by harvesting early or transferring stock to spare pens.'
        ]
    },
    'Fungal diseases Saprolegniasis': {
        'title': 'Treating Saprolegniasis / Fungal (Pond Scale)',
        'description': 'A secondary fungal infection that looks like cotton-like tufts. It attacks fish that are already stressed from netting, cold weather, or poor water.',
        'steps': [
            'Avoid any unnecessary netting, seining, or handling of the stock, as this damages their protective slime coat and invites fungal spores.',
            'Remove and properly dispose of dead or severely infected fish to prevent the fungus from releasing spores into the water.',
            'Ensure the pond depth is maintained at 1 to 1.5 meters to buffer the water against sudden temperature drops.',
            'Flush the pond with fresh water to lower the concentration of organic matter that feeds the fungus.'
        ]
    }
}

def treatment_view(request, disease_name):
    # Look up the plan based on the URL
    plan = TREATMENT_PLANS.get(disease_name)
    
    return render(request, 'detector/treatment.html', {
        'disease': disease_name,
        'plan': plan
    })