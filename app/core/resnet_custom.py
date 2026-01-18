import torch
from torchvision import models, transforms


class ResNet:
    def __init__(self, path_to_model, num_classes=7, device=None):
        self.path_to_model = path_to_model
        self.num_classes = num_classes
        self.model = self.load_model()
        self.resnet_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.converter_id_name = [
            "MEL",  # 1. Melanoma
            "NV",  # 2. Melanocytic nevi
            "BCC",  # 3. Basal cell carcinoma
            "AKIEC",  # 4. Actinic keratoses
            "BKL",  # 5. Benign keratosis-like lesions
            "DF",  # 6. Dermatofibroma
            "VASC",  # 7. Vascular lesions
        ]

      #   """
      #           MEL       0.94      0.98      0.96       123\n",
      # "          NV       0.99      0.99      0.99      1081\n",
      # "         BCC       1.00      0.97      0.98        66\n",
      # "       AKIEC       0.98      0.93      0.96        46\n",
      # "         BKL       0.97      0.97      0.97       146\n",
      # "          DF       1.00      1.00      1.00        15\n",
      # "        VASC       1.00      1.00      1.00        20\n",
      #
      #   """

    def load_model(self):
        model = models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(self.path_to_model, map_location=device))
        return model

    def predict(self, image_tensor):
        """
        Run inference on a preprocessed image tensor.
        Expects tensor shape: [1, 3, 224, 224]
        """
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            pred_class = torch.argmax(prob).item()
        return self.converter_id_name[pred_class], prob[pred_class].item()

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device

    def eval(self):
        self.model.eval()