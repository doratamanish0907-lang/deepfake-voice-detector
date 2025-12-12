import torch
import torch.nn.utils.prune as prune
from step4_train_model import VoiceCNN  # your model class

# Load original model
model = VoiceCNN()
model.load_state_dict(torch.load("voice_cnn_model.pth", map_location="cpu"))
model.eval()

print("Original model loaded.")

###############################
# 🔥 PRUNE 70% OF WEIGHTS
###############################
parameters_to_prune = []
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, "weight"))

# Remove 70% of smallest weights
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.7,
)

print("Pruning completed.")

###############################
# 🔥 APPLY DYNAMIC QUANTIZATION
###############################
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

print("Dynamic quantization completed.")

###############################
# 🔥 SAVE COMPRESSED MODEL
###############################
torch.save(model_quantized.state_dict(), "voice_cnn_model_small.pth")
print("Saved compressed model → voice_cnn_model_small.pth")
