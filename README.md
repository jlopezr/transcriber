## 🎤 Transcriber

Aplicación de consola multiplataforma para **transcribir voz en tiempo real** desde el micrófono.  
Permite elegir entre dos motores:

- **Vosk** → Rápido, offline y ligero.  
- **Whisper (faster-whisper)** → Más preciso, multilingüe y con soporte GPU/CPU.

Incluye **detección de voz (VAD)** con [`webrtcvad`](https://github.com/wiseman/py-webrtcvad), para separar frases automáticamente.

---

### 🚀 Vosk

1. Descargar un modelo de Vosk en español

Entra en 👉 https://alphacephei.com/vosk/models

Busca Spanish y descarga, por ejemplo:

- vosk-model-small-es-0.42 (ligero, rápido)
- vosk-model-es-0.42 (más grande, más preciso)

Descomprime el .zip. Te quedará una carpeta con archivos como am, conf, ivector, etc.

```bash
python .\transcribe.py --engine vosk --vosk-model .\vosk-model-es-0.42\
```

### 🚀 Whisper (faster-whisper)

```bash
python .\transcribe.py --engine whisper --whisper-model small --language es
```

Si no pone el parámetro `--language`, se intenta detecta automáticamente.

```bash
python .\transcribe.py --engine whisper --whisper-model small --language es --device cuda
```

Para forza el uso de CUDA.

## Assistant

Utiliza el modelo de lenguaje Gema-3-27B instalado localmente.

### Prerequisitos

Arrancar el servidor de Gema-3-27B con LM Studio.

```bash
# Paso 1: descargar el modelo (usa la variante cuantizada si tu hardware es limitado)
lms get gemma-3-27b-it --yes

# Paso 2: cargar el modelo en memoria con un identificador, permitiendo GPU si está disponible
lms load gemma-3-27b-it

# Paso 3: arrancar el servidor API
lms server start
```

### Uso

```bash
python .\assistant.py
```
