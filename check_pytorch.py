import torch

print("Versión de PyTorch:", torch.__version__)
print("¿Compilado con CUDA?:", torch.version.cuda)
print("¿CUDA disponible?:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU detectada:", torch.cuda.get_device_name(0))
    print("Versión de cuDNN:", torch.backends.cudnn.version())
