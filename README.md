# 🎤 Transcribe CLI

Aplicación de consola multiplataforma para **transcribir voz en tiempo real** desde el micrófono.  
Permite elegir entre dos motores:

- **Vosk** → Rápido, offline y ligero.  
- **Whisper (faster-whisper)** → Más preciso, multilingüe y con soporte GPU/CPU.

Incluye **detección de voz (VAD)** con [`webrtcvad`](https://github.com/wiseman/py-webrtcvad), para separar frases automáticamente.

---

## ✨ Características

- Multiplataforma: Linux, macOS y Windows.  
- Selección de motor (`vosk` o `whisper`).  
- Soporte para **VAD configurable** (agresividad y duración del silencio).  
- Elección de micrófono por índice.  
- Guardado opcional de las frases en archivo de texto.  
- Compatible con GPU (CUDA en NVIDIA, Metal en macOS) al usar Whisper.

## 🚀 Vosk

1. Descargar un modelo de Vosk en español

Entra en 👉 https://alphacephei.com/vosk/models

Busca Spanish y descarga, por ejemplo:

- vosk-model-small-es-0.42 (ligero, rápido)
- vosk-model-es-0.42 (más grande, más preciso)

Descomprime el .zip. Te quedará una carpeta con archivos como am, conf, ivector, etc.

```bash
python .\main.py --engine vosk --vosk-model .\vosk-model-es-0.42\
```

## 🚀 Whisper (faster-whisper)

```bash
python .\main.py --engine whisper --whisper-model small --language es
```

Si no pone el parámetro `--language`, se intenta detecta automáticamente.

```bash
python .\main.py --engine whisper --whisper-model small --language es --device cuda
```

Para forza el uso de CUDA.

