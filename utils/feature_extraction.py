import warnings
warnings.filterwarnings("ignore")
import librosa
import numpy as np

def extract_features(wav_path, transcript):
    y, sr = librosa.load(wav_path)

    # Pauses
    intervals = librosa.effects.split(y, top_db=30)
    pauses = []
    for i in range(1, len(intervals)):
        pause = librosa.get_duration(y=y[intervals[i-1][1]:intervals[i][0]], sr=sr)
        if pause > 0.2:
            pauses.append(pause)

    # Speech Rate
    duration = librosa.get_duration(y=y, sr=sr)
    words = len(transcript.split())
    speech_rate = words / duration if duration > 0 else 0

    # Hesitations
    hesitations = ['uh', 'um']
    hesitation_count = sum(transcript.lower().count(h) for h in hesitations)

    # Pitch Variability
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_var = np.std(pitch_values) if len(pitch_values) > 0 else 0

    return {
        "pauses": len(pauses),
        "speech_rate": speech_rate,
        "hesitations": hesitation_count,
        "pitch_var": pitch_var
    }
