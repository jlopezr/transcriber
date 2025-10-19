#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from pathlib import Path

# --- Coqui TTS (opcional, solo si se usa engine=coqui) ---
def synthesize_coqui(text, out_path, model_name, speaker=None, speed=1.0):
    try:
        from TTS.api import TTS
    except ImportError:
        print("ERROR: Falta el paquete 'TTS'. Instala con: pip install TTS soundfile", file=sys.stderr)
        sys.exit(1)

    # Carga el modelo (ejemplos de modelos con español)
    # - 'tts_models/es/css10/vits'  -> voz español (dataset CSS10)
    # - También puedes usar modelos multilingües si soportan ES.
    tts = TTS(model_name=model_name, progress_bar=False, gpu=False)

    # Algunos modelos aceptan speaker o speaker_idx; si no aplica, lo ignora.
    # speed ~1.0 es normal (no todos los modelos soportan rate).
    try:
        tts.tts_to_file(
            text=text,
            file_path=out_path,
            speaker=speaker,
            speed=speed
        )
    except TypeError:
        # Fallback si el modelo no soporta speed/speaker
        tts.tts_to_file(
            text=text,
            file_path=out_path
        )


# --- Piper (vía binario) ---
def synthesize_piper(text, out_path, model_path, config_path=None, speaker=None, length_scale=1.0):
    """
    length_scale <1.0 = habla más rápida, >1.0 = más lenta (en Piper).
    """
    # Busca el ejecutable 'piper' o 'piper.exe' en PATH o en el directorio actual
    piper_exe = "piper.exe" if os.name == "nt" else "piper"
    if not shutil_which(piper_exe):
        # intenta en el directorio del script
        local_exe = Path(__file__).parent / piper_exe
        if local_exe.exists():
            piper_exe = str(local_exe)
        else:
            print("ERROR: No encuentro el ejecutable de Piper. Añádelo al PATH o déjalo junto al script.", file=sys.stderr)
            sys.exit(1)

    if not Path(model_path).exists():
        print(f"ERROR: No encuentro el modelo de Piper: {model_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [piper_exe, "--model", model_path, "--output_file", out_path]
    if config_path:
        if not Path(config_path).exists():
            print(f"ADVERTENCIA: No encuentro config JSON: {config_path}. Continúo sin --config.", file=sys.stderr)
        else:
            cmd += ["--config", config_path]

    # Control de velocidad (en Piper se controla con length_scale; menor = más rápido)
    if length_scale and float(length_scale) != 1.0:
        cmd += ["--length_scale", str(length_scale)]

    if speaker is not None:
        # Para voces multi-speaker: suele ser un índice entero o nombre, según el modelo.
        cmd += ["--speaker", str(speaker)]

    # Pasamos el texto por stdin
    try:
        proc = subprocess.run(cmd, input=text.encode("utf-8"), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("ERROR ejecutando Piper:", e.stderr.decode("utf-8", errors="ignore"), file=sys.stderr)
        sys.exit(1)


def shutil_which(cmd):
    # Minimalista para evitar dependencia de shutil en versiones viejas
    from shutil import which
    return which(cmd)


def main():
    parser = argparse.ArgumentParser(description="TTS en español con Coqui o Piper (local/offline).")
    parser.add_argument("--engine", choices=["coqui", "piper"], required=True, help="Motor TTS a usar.")
    parser.add_argument("--text", help="Texto a sintetizar. Si no se pasa, usa --file.")
    parser.add_argument("--file", help="Ruta a archivo de texto.")
    parser.add_argument("--out", default="salida.wav", help="Archivo de salida .wav (por defecto: salida.wav)")

    # Coqui params
    parser.add_argument("--coqui-model", default="tts_models/es/css10/vits", help="Nombre del modelo Coqui (p.ej., tts_models/es/css10/vits)")
    parser.add_argument("--coqui-speaker", default=None, help="Altavoz (si el modelo lo soporta).")
    parser.add_argument("--coqui-speed", type=float, default=1.0, help="Velocidad (si el modelo la soporta). 1.0 = normal")

    # Piper params
    parser.add_argument("--piper-model", help="Ruta al modelo Piper (.onnx)", default=None)
    parser.add_argument("--piper-config", help="Ruta al config .json de Piper (opcional).", default=None)
    parser.add_argument("--piper-speaker", help="Índice/nombre de altavoz para modelos multi-speaker.", default=None)
    parser.add_argument("--piper-length-scale", type=float, default=1.0, help="<1.0 más rápido, >1.0 más lento")

    args = parser.parse_args()

    # Leer texto
    if not args.text and not args.file:
        print("ERROR: Debes proporcionar --text o --file.", file=sys.stderr)
        sys.exit(1)
    if args.file:
        try:
            text = Path(args.file).read_text(encoding="utf-8")
        except Exception as e:
            print(f"ERROR leyendo {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        text = args.text

    # Asegurar carpeta de salida
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.engine == "coqui":
        synthesize_coqui(
            text=text,
            out_path=str(out_path),
            model_name=args.coqui_model,
            speaker=args.coqui_speaker,
            speed=args.coqui_speed
        )
    else:
        if not args.piper_model:
            print("ERROR: Para Piper debes pasar --piper-model con la ruta al .onnx", file=sys.stderr)
            sys.exit(1)
        synthesize_piper(
            text=text,
            out_path=str(out_path),
            model_path=args.piper_model,
            config_path=args.piper_config,
            speaker=args.piper_speaker,
            length_scale=args.piper_length_scale
        )

    print(f"✅ Audio generado: {out_path.resolve()}")


if __name__ == "__main__":
    main()
