#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import signal
import queue
import time
import json
import math
import numpy as np
import sounddevice as sd
import webrtcvad

SAMPLE_RATE = 16000
FRAME_MS = 30  # 10, 20 o 30 ms; 30 ms suele ir bien
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 a 16 kHz
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # int16 = 2 bytes

stop_flag = False
def handle_sigint(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, handle_sigint)

def list_devices_and_exit():
    print(sd.query_devices())
    sys.exit(0)

def write_out(fpath, text):
    if not fpath:
        return
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

def init_vosk(model_path):
    try:
        from vosk import Model
    except ImportError:
        print("Falta 'vosk'. Instala con: pip install vosk", file=sys.stderr)
        sys.exit(1)
    if not model_path or not os.path.isdir(model_path):
        print("Debes proporcionar --vosk-model con la carpeta del modelo.", file=sys.stderr)
        sys.exit(1)
    return Model(model_path)

def init_whisper(whisper_model, device, compute_type):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Falta 'faster-whisper'. Instala con: pip install faster-whisper", file=sys.stderr)
        sys.exit(1)
    return WhisperModel(whisper_model, device=device, compute_type=compute_type)

class VADSegmenter:
    """
    Segmenta audio en trozos de habla usando WebRTC VAD.
    Alimenta frames de 30ms (bytes int16 mono 16k).
    Cuando detecta silencio prolongado (silence_ms), emite un segmento.
    """
    def __init__(self, aggressiveness=2, silence_ms=800):
        self.vad = webrtcvad.Vad(int(aggressiveness))
        self.silence_frames_needed = max(1, int(round(silence_ms / FRAME_MS)))
        self.in_speech = False
        self.silence_count = 0
        self.buffer = bytearray()

    def process_frame(self, frame_bytes):
        """Devuelve un bytes de segmento cuando se cierra; si no, None."""
        is_voice = self.vad.is_speech(frame_bytes, SAMPLE_RATE)

        if is_voice:
            if not self.in_speech:
                # transición a habla
                self.in_speech = True
                self.buffer = bytearray()
                self.silence_count = 0
            self.buffer.extend(frame_bytes)
            self.silence_count = 0
            return None
        else:
            if self.in_speech:
                self.silence_count += 1
                self.buffer.extend(frame_bytes)  # añadimos un poco de cola
                if self.silence_count >= self.silence_frames_needed:
                    # fin del segmento
                    seg = bytes(self.buffer)
                    self.in_speech = False
                    self.buffer = bytearray()
                    self.silence_count = 0
                    return seg
            # fuera de habla: nada
            return None

    def flush(self):
        if self.in_speech and self.buffer:
            seg = bytes(self.buffer)
            self.in_speech = False
            self.buffer = bytearray()
            self.silence_count = 0
            return seg
        return None

def transcribe_with_vosk(model, pcm_bytes):
    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)
    # Alimentamos en bloques para evitar límites de tamaño
    # aunque normalmente un AcceptWaveform único bastaría
    for i in range(0, len(pcm_bytes), 4000):
        rec.AcceptWaveform(pcm_bytes[i:i+4000])
    try:
        j = json.loads(rec.FinalResult())
        return j.get("text", "").strip()
    except json.JSONDecodeError:
        return ""

def transcribe_with_whisper(model, pcm_bytes, language="auto"):
    # Convertimos a float32 -1..1
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # Nota: faster-whisper acepta arrays numpy directamente
    # Config: sin timestamps para mostrar texto sencillo
    segments, info = model.transcribe(
        audio,
        language=None if language == "auto" else language,
        vad_filter=False,  # ya usamos VAD externo
        word_timestamps=False,
        beam_size=5,
        best_of=5
    )
    # 'segments' es un generador; concatenamos
    text_parts = []
    for s in segments:
        # s.text ya viene con espacios adecuados
        text_parts.append(s.text.strip())
    return " ".join(tp for tp in text_parts if tp)

def main():
    parser = argparse.ArgumentParser(description="Transcripción por voz en consola con VAD (Vosk o Whisper).")
    parser.add_argument("--engine", choices=["vosk", "whisper"], required=False, default="vosk",
                        help="Motor de reconocimiento.")
    parser.add_argument("--vosk-model", type=str, help="Carpeta del modelo Vosk.")
    parser.add_argument("--whisper-model", type=str, default="small",
                        help="Tamaño/nombre del modelo faster-whisper (p.ej. tiny, base, small, medium, large-v3).")
    parser.add_argument("--language", type=str, default="auto",
                        help="Idioma para Whisper (es, en, auto...). Vosk depende del modelo elegido.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Dispositivo para Whisper: auto|cpu|cuda|metal.")
    parser.add_argument("--compute-type", type=str, default="int8_float16",
                        help="Tipo de cómputo Whisper (p.ej. float16, int8, int8_float16).")
    parser.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0,1,2,3],
                        help="0=suave, 3=muy estricto.")
    parser.add_argument("--silence-ms", type=int, default=800,
                        help="Silencio necesario para cerrar una frase.")
    parser.add_argument("--input-device", type=int, default=None,
                        help="Índice del micrófono a usar (ver --list-devices).")
    parser.add_argument("--list-devices", action="store_true", help="Lista dispositivos de audio y sale.")
    parser.add_argument("--out", type=str, default=None, help="Guardar frases finalizadas en este archivo.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices_and_exit()

    # Inicializamos motor
    if args.engine == "vosk":
        model = init_vosk(args.vosk_model)
        engine_fn = lambda pcm: transcribe_with_vosk(model, pcm)
        engine_name = f"Vosk ({os.path.basename(args.vosk_model.rstrip(os.sep)) if args.vosk_model else ''})"
    else:
        wmodel = init_whisper(args.whisper_model, args.device, args.compute_type)
        engine_fn = lambda pcm: transcribe_with_whisper(wmodel, pcm, language=args.language)
        engine_name = f"Whisper ({args.whisper_model}, device={args.device}, compute={args.compute_type}, lang={args.language})"

    # Cola de audio y callback
    audio_q = queue.Queue(maxsize=50)
    leftover = bytearray()
    seg = VADSegmenter(aggressiveness=args.vad_aggressiveness, silence_ms=args.silence_ms)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        try:
            audio_q.put_nowait(bytes(indata))
        except queue.Full:
            # Si vamos por detrás, descartamos el bloque más viejo
            _ = audio_q.get_nowait()
            audio_q.put_nowait(bytes(indata))

    # Abrimos el stream de entrada a 16 kHz mono int16
    print(f"\nMotor: {engine_name}")
    print("Escuchando micrófono (Ctrl+C para salir)...\n")
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=SAMPLES_PER_FRAME,  # 30 ms
            dtype="int16",
            channels=1,
            callback=audio_callback,
            device=args.input_device
        ):
            last_print_partial = 0.0
            while not stop_flag:
                try:
                    chunk = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                leftover.extend(chunk)
                # Procesamos en frames exactos de 30ms para VAD
                while len(leftover) >= BYTES_PER_FRAME:
                    frame = bytes(leftover[:BYTES_PER_FRAME])
                    del leftover[:BYTES_PER_FRAME]
                    maybe_seg = seg.process_frame(frame)
                    if maybe_seg:
                        text = engine_fn(maybe_seg)
                        if text:
                            print(f"→ {text}")
                            write_out(args.out, text)
                        else:
                            # Nada confiable, no imprimimos
                            pass

            # flush final
            final_seg = seg.flush()
            if final_seg:
                text = engine_fn(final_seg)
                if text:
                    print(f"→ {text}")
                    write_out(args.out, text)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

