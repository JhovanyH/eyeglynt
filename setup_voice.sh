#!/bin/bash
# One-time setup of EyeGlynt's voice (Piper) on the Pi. In the Terminal:
#     cd ~/eyeglynt
#     bash setup_voice.sh
# It needs internet (about 100 MB to download) and takes a few minutes.
#
# To get a different voice (hear them at https://rhasspy.github.io/piper-samples/),
# add its name, e.g.:
#     bash setup_voice.sh en_US-ryan-medium
# and then set VOICE_NAME in main.py to the same name.

set -e                                        # stop at the first error
cd "$(dirname "$0")"                          # work inside the eyeglynt folder
VOICE="${1:-en_US-lessac-medium}"             # the voice to download
ENV="${EYEGLYNT_ENV:-$HOME/eyeglynt_env}"     # the app's Python environment

if [ ! -x "$ENV/bin/python" ]; then
    echo "Could not find the app's Python environment at $ENV"
    exit 1
fi

echo "1/3 Installing Piper into $ENV ..."
# --no-deps leaves numpy and protobuf exactly as MediaPipe needs them.
"$ENV/bin/pip" install --no-deps piper-tts onnxruntime pathvalidate

echo "2/3 Downloading the voice $VOICE ..."
"$ENV/bin/python" -m piper.download_voices --download-dir voices "$VOICE"

echo "3/3 Testing the voice, listen to the speaker ..."
"$ENV/bin/python" -m piper -m "voices/$VOICE.onnx" -f /tmp/eyeglynt_voice_test.wav \
    --length-scale 1.15 -- "Hello, I am EyeGlynt. My new voice is ready."
aplay -q /tmp/eyeglynt_voice_test.wav

echo
echo "Done. If you heard the voice, start the app as usual."
