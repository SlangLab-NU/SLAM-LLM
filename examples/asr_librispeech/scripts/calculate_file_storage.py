import os
import librosa

def get_total_audio_duration(folder):
    total_duration = 0.0
    audio_file_count = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(('.wav', '.flac')):
                filepath = os.path.join(dirpath, filename)
                # Skip symbolic links to avoid counting files twice
                if not os.path.islink(filepath):
                    try:
                        # Get the duration of the audio file
                        duration = librosa.get_duration(path=filepath)
                        total_duration += duration
                        audio_file_count += 1
                        # Print a message every 10,000 files processed
                        if audio_file_count % 100 == 0:
                            print(f"Processed {audio_file_count} files so far...")
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
    return total_duration, audio_file_count

folder_path = "/work/van-speech-nlp/jindaznb/jslpnb/mllm_experiments/aphasia/data_processed/audios"
print("starting")
total_duration_seconds, audio_count = get_total_audio_duration(folder_path)
total_duration_hours = total_duration_seconds / 3600

print(f"Total number of audio files: {audio_count}")
print(f"Total duration: {total_duration_seconds:.2f} seconds ({total_duration_hours:.2f} hours)")
