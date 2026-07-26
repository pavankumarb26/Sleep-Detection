import wave
import struct
import math

def generate_beep(filename="alarm.wav", duration=1.5, frequency=1000.0, sample_rate=44100.0):
    """
    Generates a simple high-pitched beep WAV file to act as the alarm sound.
    """
    num_samples = int(duration * sample_rate)
    amplitude = 16384  # Moderate volume (max is 32767 for 16-bit signed integer)
    
    # Open WAV file for writing
    with wave.open(filename, 'wb') as wav_file:
        # parameters: nchannels (1 for mono), sampwidth (2 bytes for 16-bit), framerate, nframes, comptype, compname
        wav_file.setparams((1, 2, int(sample_rate), num_samples, 'NONE', 'not compressed'))
        
        # Write sine wave values
        for i in range(num_samples):
            # Calculate the sine wave value at this time step
            t = float(i) / sample_rate
            value = int(amplitude * math.sin(2.0 * math.pi * frequency * t))
            
            # Pack the 16-bit integer as binary data (little endian short: '<h')
            data = struct.pack('<h', value)
            wav_file.writeframes(data)
            
    print(f"Generated synthetic alarm file: {filename}")

if __name__ == "__main__":
    generate_beep()
