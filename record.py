import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os


output_dir = "data/voice"
os.makedirs(output_dir, exist_ok=True)

fs = 16000      #16khz rate 
duration = 3    #seconds
quantity = 20   #number of file recording

print("🎤 Recording is starting")

for r in range(quantity):
	print(f"Recording... {r+1}/{quantity}...")

	audio = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='int16')
	sd.wait()

	filename = os.path.join(output_dir, f"voice_{r+1:02d}.wav")
	wav.write(filename, fs, audio)
	print(f"Saved: {filename}")
print("Done recording 20 files.")
