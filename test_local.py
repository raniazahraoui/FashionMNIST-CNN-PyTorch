"""
Script d'Inférence Locale pour Fashion-MNIST CNN
================================================

Ce script permet d'utiliser le modèle entraîné pour faire des prédictions
sur de nouvelles images en local.

Prérequis:
- fashion_mnist_cnn_final.pth (modèle téléchargé depuis Colab)
- PyTorch installé: pip install torch torchvision pillow matplotlib numpy
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# 1. DÉFINITION DE L'ARCHITECTURE DU MODÈLE
# ============================================================================

class FashionCNN(nn.Module):
    """
    Architecture CNN identique à celle utilisée pour l'entraînement
    """
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        # Couches convolutionnelles
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Couches fully-connected
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Bloc 1
        x = self.pool1(self.relu1(self.conv1(x)))
        # Bloc 2
        x = self.pool2(self.relu2(self.conv2(x)))
        # Fully-connected
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        return self.fc2(x)

# ============================================================================
# 2. CHARGEMENT DU MODÈLE
# ============================================================================

def load_model(model_path='fashion_mnist_cnn_final.pth'):
    """
    Charge le modèle entraîné depuis le fichier .pth
    
    Args:
        model_path: Chemin vers le fichier du modèle
    
    Returns:
        model: Modèle chargé et prêt pour l'inférence
        device: Device utilisé (cuda ou cpu)
    """
    # Vérifier si le fichier existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Fichier modèle non trouvé: {model_path}\n"
            f"Assurez-vous d'avoir téléchargé le modèle depuis Colab."
        )
    
    # Détection du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé: {device}")
    
    # Instanciation du modèle
    model = FashionCNN().to(device)
    
    # Chargement des poids
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Mode évaluation (désactive dropout, etc.)
    model.eval()
    
    print(f"✓ Modèle chargé depuis: {model_path}")
    
    # Afficher les infos du checkpoint si disponibles
    if 'best_acc' in checkpoint:
        print(f"✓ Précision du modèle: {checkpoint['best_acc']:.2f}%")
    
    return model, device

# ============================================================================
# 3. CLASSES FASHION-MNIST
# ============================================================================

CLASS_NAMES = [
    'T-shirt/top',    # 0
    'Trouser',        # 1
    'Pullover',       # 2
    'Dress',          # 3
    'Coat',           # 4
    'Sandal',         # 5
    'Shirt',          # 6
    'Sneaker',        # 7
    'Bag',            # 8
    'Ankle boot'      # 9
]

# ============================================================================
# 4. PRÉTRAITEMENT DES IMAGES
# ============================================================================

# Transformations identiques à l'entraînement
transform = transforms.Compose([
    transforms.Grayscale(),              # Convertir en niveaux de gris
    transforms.Resize((28, 28)),         # Redimensionner à 28x28
    transforms.ToTensor(),               # Convertir en tensor
    transforms.Normalize((0.5,), (0.5,)) # Normaliser
])

# ============================================================================
# 5. FONCTION DE PRÉDICTION
# ============================================================================

def predict_image(model, image_path, device):
    """
    Prédit la classe d'une image
    
    Args:
        model: Modèle PyTorch
        image_path: Chemin vers l'image
        device: Device (cuda/cpu)
    
    Returns:
        prediction: Nom de la classe prédite
        confidence: Confiance (probabilité) en %
        all_probs: Probabilités pour toutes les classes
    """
    # Vérifier si l'image existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image non trouvée: {image_path}")
    
    # Charger et prétraiter l'image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
    
    prediction = CLASS_NAMES[pred_idx]
    all_probs = probs[0].cpu().numpy() * 100
    
    return prediction, confidence, all_probs

# ============================================================================
# 6. VISUALISATION DES RÉSULTATS
# ============================================================================

def visualize_prediction(image_path, prediction, confidence, all_probs):
    """
    Affiche l'image avec la prédiction et les probabilités
    
    Args:
        image_path: Chemin de l'image
        prediction: Classe prédite
        confidence: Confiance en %
        all_probs: Probabilités pour toutes les classes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Affichage de l'image
    image = Image.open(image_path).convert('L')
    ax1.imshow(image, cmap='gray')
    ax1.set_title(
        f"Prédiction: {prediction}\nConfiance: {confidence:.2f}%",
        fontsize=14,
        fontweight='bold',
        color='green' if confidence > 80 else 'orange'
    )
    ax1.axis('off')
    
    # Graphique des probabilités
    y_pos = np.arange(len(CLASS_NAMES))
    colors = ['green' if i == CLASS_NAMES.index(prediction) else 'lightblue' 
              for i in range(len(CLASS_NAMES))]
    
    ax2.barh(y_pos, all_probs, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(CLASS_NAMES)
    ax2.set_xlabel('Probabilité (%)', fontsize=12)
    ax2.set_title('Distribution des Probabilités', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Ajouter les valeurs sur les barres
    for i, prob in enumerate(all_probs):
        if prob > 2:  # Afficher seulement si > 2%
            ax2.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. PRÉDICTION SUR UNE IMAGE DU DATASET DE TEST
# ============================================================================

def predict_from_test_dataset(model, device, index=0):
    """
    Prédit une image du dataset de test Fashion-MNIST
    
    Args:
        model: Modèle PyTorch
        device: Device utilisé
        index: Index de l'image dans le dataset (0-9999)
    """
    from torchvision import datasets
    
    # Charger le dataset de test
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Récupérer l'image et le label
    image, true_label = test_dataset[index]
    image_display = image.squeeze() * 0.5 + 0.5  # Dénormalisation
    
    # Prédiction
    with torch.no_grad():
        image_input = image.unsqueeze(0).to(device)
        output = model(image_input)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
        all_probs = probs[0].cpu().numpy() * 100
    
    prediction = CLASS_NAMES[pred_idx]
    true_class = CLASS_NAMES[true_label]
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Image
    ax1.imshow(image_display, cmap='gray')
    is_correct = pred_idx == true_label
    color = 'green' if is_correct else 'red'
    status = '✓ CORRECT' if is_correct else '✗ INCORRECT'
    
    ax1.set_title(
        f"Vérité: {true_class}\n"
        f"Prédiction: {prediction}\n"
        f"Confiance: {confidence:.2f}% {status}",
        fontsize=12,
        fontweight='bold',
        color=color
    )
    ax1.axis('off')
    
    # Probabilités
    y_pos = np.arange(len(CLASS_NAMES))
    colors = ['green' if i == pred_idx else 'red' if i == true_label else 'lightblue' 
              for i in range(len(CLASS_NAMES))]
    
    ax2.barh(y_pos, all_probs, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(CLASS_NAMES)
    ax2.set_xlabel('Probabilité (%)', fontsize=12)
    ax2.set_title('Distribution des Probabilités', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.show()
    
    return is_correct, prediction, confidence

# ============================================================================
# 8. FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale pour tester le modèle
    """
    print("="*70)
    print("PRÉDICTION AVEC FASHION-MNIST CNN")
    print("="*70)
    
    # Charger le modèle
    try:
        model, device = load_model('fashion_mnist_cnn_final.pth')
    except FileNotFoundError as e:
        print(f"\n❌ Erreur: {e}")
        return
    
    print("\n" + "="*70)
    print("OPTIONS DE TEST")
    print("="*70)
    print("1. Prédire une image personnelle")
    print("2. Tester sur des images du dataset de test")
    print("="*70)
    
    choice = input("\nVotre choix (1 ou 2): ").strip()
    
    if choice == '1':
        # Prédiction sur image personnelle
        image_path = input("Chemin de l'image: ").strip()
        
        try:
            prediction, confidence, all_probs = predict_image(model, image_path, device)
            
            print("\n" + "="*70)
            print("RÉSULTAT DE LA PRÉDICTION")
            print("="*70)
            print(f"Classe prédite: {prediction}")
            print(f"Confiance: {confidence:.2f}%")
            print("\nTop 3 des prédictions:")
            top3_indices = np.argsort(all_probs)[::-1][:3]
            for i, idx in enumerate(top3_indices, 1):
                print(f"  {i}. {CLASS_NAMES[idx]}: {all_probs[idx]:.2f}%")
            
            visualize_prediction(image_path, prediction, confidence, all_probs)
            
        except Exception as e:
            print(f"\n❌ Erreur lors de la prédiction: {e}")
    
    elif choice == '2':
        # Test sur dataset
        print("\nTest sur 5 images aléatoires du dataset de test...")
        
        correct = 0
        total = 5
        
        for i in range(total):
            random_index = np.random.randint(0, 10000)
            print(f"\n--- Image {i+1}/{total} (Index: {random_index}) ---")
            
            is_correct, pred, conf = predict_from_test_dataset(model, device, random_index)
            if is_correct:
                correct += 1
        
        print(f"\n{'='*70}")
        print(f"Résultat: {correct}/{total} prédictions correctes ({100*correct/total:.1f}%)")
        print(f"{'='*70}")
    
    else:
        print("\n❌ Choix invalide!")

# ============================================================================
# EXÉCUTION DU SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# EXEMPLES D'UTILISATION AVANCÉE
# ============================================================================

"""
# Exemple 1: Prédiction simple
model, device = load_model('fashion_mnist_cnn_final.pth')
prediction, confidence, probs = predict_image(model, 'mon_tshirt.jpg', device)
print(f"Prédiction: {prediction} ({confidence:.2f}%)")

# Exemple 2: Prédictions en batch
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
for path in image_paths:
    pred, conf, _ = predict_image(model, path, device)
    print(f"{path}: {pred} ({conf:.2f}%)")

# Exemple 3: Tester plusieurs images du dataset
model, device = load_model()
for idx in [0, 100, 500, 1000, 5000]:
    predict_from_test_dataset(model, device, idx)
"""