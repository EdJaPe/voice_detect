import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os


output_dir1 = "data/voice"
output_dir2 = "data/silence"
output_dir3 = "data/background"

#manually changing the directory
os.makedirs(output_dir3, exist_ok=True)

fs = 16000      #16khz rate 
duration = 3    #seconds
quantity = 20   #number of file recording

print("🎤 Recording is starting")

for r in range(quantity):
	print(f"Recording... {r+1}/{quantity}...")

	#setting the recording settings
	audio = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='int16')
	sd.wait()

	filename = os.path.join(output_dir3, f"background_{r+1:02d}.wav")
	wav.write(filename, fs, audio)
	print(f"Saved: {filename}")
print("Done recording 20 files.")
