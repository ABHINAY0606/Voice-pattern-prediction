from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import whisper
import os
import tempfile
import wave
from utils.feature_extraction import extract_features  # You must have this module

app = Flask(__name__)
model = whisper.load_model("base")

AUDIO_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/analyze', methods=['POST'])
def analyze_multiple():
    files = request.files.getlist("files")

    if not files or len(files) == 0:
        return render_template("results.html", results=[{"filename": "None", "error": "No files uploaded"}])

    results = []

    for file in files[:10]:  # Limit to 10 files
        if not (file.filename.endswith(".mp3") or file.filename.endswith(".wav")):
            results.append({
                "filename": file.filename,
                "error": "Unsupported file format"
            })
            continue

        ext = os.path.splitext(file.filename)[-1].lower()

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            file.save(tmp_file.name)
            tmp_file.close()

            wav_path = tmp_file.name.replace(ext, ".wav")

            try:
                # Convert to WAV
                if ext == ".mp3":
                    sound = AudioSegment.from_mp3(tmp_file.name)
                    sound.export(wav_path, format="wav")
                elif ext == ".wav":
                    sound = AudioSegment.from_wav(tmp_file.name)
                    sound.export(wav_path, format="wav")
                else:
                    continue

                # Audio file info
                with wave.open(wav_path, 'rb') as wf:
                    num_channels = wf.getnchannels()
                    framerate = wf.getframerate()
                    duration = wf.getnframes() / framerate

                # Transcription
                result = model.transcribe(wav_path)
                transcript = result["text"]

                # Feature Extraction
                features = extract_features(wav_path, transcript)

                # Risk Score
                risk_score = compute_risk_score(features)

                results.append({
                    "filename": file.filename,
                    "channels": num_channels,
                    "framerate": framerate,
                    "duration": round(duration, 2),
                    "transcript": transcript,
                    "features": features,
                    "risk_score": risk_score
                })

            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })

            finally:
                try:
                    os.remove(tmp_file.name)
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                except Exception as cleanup_error:
                    print(f"Error deleting temp file: {cleanup_error}")

    return render_template("results.html", results=results)

def compute_risk_score(features):
    # Simple heuristic scoring
    score = (
        features.get("pauses", 0) * 0.3 +
        features.get("hesitations", 0) * 0.3 +
        max(0, 1.5 - features.get("speech_rate", 0)) * 0.2 +
        features.get("pitch_var", 0) * 0.2
    )
    return round(score, 2)

if __name__ == '__main__':
    app.run(debug=True)
