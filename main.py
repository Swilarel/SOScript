import cv2
import numpy as np
import subprocess
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import re

def get_roi(frame, height_ratio=0.25, width_ratio=0.25, grayscale=True):
    """
    Extracts the ROI from the frame.
    Converts to grayscale if specified.
    """
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame.shape[:2]
    roi = frame[int(height * (1 - height_ratio)):height, 0:int(width * width_ratio)]
    return roi

def compute_baseline_and_detect_start(cap, num_baseline_frames=10, threshold=30, match_ratio=0.5, progress_callback=None):
    """
    Computes the baseline from the first few frames and detects the start time
    where the reaction begins based on ROI differences.

    Returns:
        start_time (float): The timestamp in seconds where the reaction starts.
                             Returns None if not found.
    """
    baselines = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    for i in range(num_baseline_frames):
        ret, frame = cap.read()
        if not ret:
            break
        roi = get_roi(frame)
        baselines.append(roi.astype(np.float32))
        if progress_callback:
            progress_callback(i / (num_baseline_frames + 100))  # Minimal progress update

    if not baselines:
        raise ValueError("No frames read from the video for baseline computation.")

    baseline = np.mean(baselines, axis=0)

    frame_count = num_baseline_frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            return None  # Reaction start not found
        roi = get_roi(frame)
        diff = cv2.absdiff(baseline.astype(np.uint8), roi)
        non_zero_count = np.count_nonzero(diff > threshold)
        if non_zero_count > match_ratio * roi.size:
            start_time = frame_count / fps
            return start_time
        frame_count += 1
        if progress_callback:
            progress_callback((frame_count) / total_frames)

def get_video_duration(video_path):
    """
    Retrieves the duration of the video in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError(f"FPS is zero for video: {video_path}")
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration

def trim_video_ffmpeg(input_path, output_path, start_time, progress_callback=None):
    """
    Trims the video starting from the specified start_time using ffmpeg.
    This method is faster than using MoviePy.
    """
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-ss', str(start_time),
        '-i', input_path,
        '-c', 'copy',  # Copy codec to avoid re-encoding
        output_path
    ]

    try:
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Optionally, parse ffmpeg progress here
                pass
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, ffmpeg_command)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr}")

def is_ffmpeg_encoder_available(encoder_name):
    """
    Checks if the specified FFmpeg encoder is available.
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        encoders = result.stdout.lower()
        return encoder_name.lower() in encoders
    except Exception as e:
        print(f"Error checking FFmpeg encoders: {e}")
        return False

def merge_videos_ffmpeg(main_video_path, synced_reaction_path, output_path, progress_callback=None, encoder_callback=None):
    """
    Merges the main video and synchronized reaction video side by side using ffmpeg.
    Utilizes hardware acceleration if available.
    """
    # Ensure input files exist
    if not os.path.isfile(main_video_path):
        raise FileNotFoundError(f"Main video not found: {main_video_path}")
    if not os.path.isfile(synced_reaction_path):
        raise FileNotFoundError(f"Synchronized reaction video not found: {synced_reaction_path}")

    # Get the duration of the main video to calculate progress
    total_duration = get_video_duration(main_video_path)

    # Determine the encoder to use
    if is_ffmpeg_encoder_available('h264_nvenc'):
        video_codec = 'h264_nvenc'
        encoder_description = 'NVIDIA H.264 Encoder (h264_nvenc)'
        using_cpu_encoding = False
    elif is_ffmpeg_encoder_available('libx264'):
        video_codec = 'libx264'
        encoder_description = 'Software H.264 Encoder (libx264)'
        using_cpu_encoding = True
    else:
        raise RuntimeError("Neither 'h264_nvenc' nor 'libx264' encoders are available in FFmpeg.")

    # Inform the user about the encoder being used
    if encoder_callback:
        encoder_callback(encoder_description)

    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output files without asking
        '-i', main_video_path,
        '-i', synced_reaction_path,
        '-filter_complex',
        "[1:a]volume=1.5[a1];"  # Increase volume of reaction audio
        "[0:v][1:v]hstack=inputs=2[v];"  # Stack videos side by side
        "[0:a][a1]amix=inputs=2[a]",  # Mix audio streams
        '-map', '[v]',
        '-map', '[a]',
        '-c:v', video_codec,  # Use determined video encoder
        '-preset', 'fast' if not using_cpu_encoding else 'medium',
        '-b:v', '5M',
        '-c:a', 'aac',
        '-b:a', '192k',
        output_path
    ]

    # Adjust preset for libx264 if necessary
    if video_codec == 'libx264':
        # Optionally, add more libx264 specific settings
        pass

    time_pattern = re.compile(r'time=(\d+):(\d+):(\d+\.\d+)')

    try:
        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Parse FFmpeg progress
                match = time_pattern.search(output)
                if match:
                    hours, minutes, seconds = match.groups()
                    current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                    # Calculate progress percentage
                    percentage = (current_time / total_duration) * 100
                    percentage = min(percentage, 100)
                    if progress_callback:
                        progress_callback(percentage)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, ffmpeg_command)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr}")

def synchronize_videos(reaction_video_path, synced_reaction_path, num_baseline_frames=10, threshold=30, match_ratio=0.5, progress_callback=None):
    """
    Synchronizes the reaction video by trimming it based on detected preview start.
    """
    cap = cv2.VideoCapture(reaction_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open reaction video: {reaction_video_path}")

    try:
        start_time = compute_baseline_and_detect_start(
            cap,
            num_baseline_frames=num_baseline_frames,
            threshold=threshold,
            match_ratio=match_ratio,
            progress_callback=progress_callback
        )
    finally:
        cap.release()

    if start_time is not None:
        trim_video_ffmpeg(reaction_video_path, synced_reaction_path, start_time, progress_callback)
    else:
        raise ValueError("Reaction start not detected in the reaction video.")

def delete_file(file_path):
    """
    Deletes the specified file if it exists.
    """
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Failed to delete temporary file {file_path}: {e}")
        raise

class VideoSyncMergerApp:
    def __init__(self, master):
        self.master = master
        master.title("SOScript")
        master.geometry("525x400")  # Increased height to accommodate new labels
        master.resizable(False, False)

        # Initialize variables
        self.main_video_path = tk.StringVar()
        self.reaction_video_path = tk.StringVar()
        self.output_dir_path = tk.StringVar()
        self.output_filename = tk.StringVar(value="merged_output.mp4")

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        padding = {'padx': 10, 'pady': 10}

        # Select Main Video
        self.main_video_label = tk.Label(self.master, text="Anime Episode:")
        self.main_video_label.grid(row=0, column=0, sticky='e', **padding)

        self.main_video_entry = tk.Entry(self.master, textvariable=self.main_video_path, width=40, state='readonly')
        self.main_video_entry.grid(row=0, column=1, **padding)

        self.main_video_button = tk.Button(self.master, text="Browse", command=self.browse_main_video)
        self.main_video_button.grid(row=0, column=2, **padding)

        # Select Reaction Video
        self.reaction_video_label = tk.Label(self.master, text="Reaction Video:")
        self.reaction_video_label.grid(row=1, column=0, sticky='e', **padding)

        self.reaction_video_entry = tk.Entry(self.master, textvariable=self.reaction_video_path, width=40, state='readonly')
        self.reaction_video_entry.grid(row=1, column=1, **padding)

        self.reaction_video_button = tk.Button(self.master, text="Browse", command=self.browse_reaction_video)
        self.reaction_video_button.grid(row=1, column=2, **padding)

        # Select Output Directory
        self.output_label = tk.Label(self.master, text="Output Directory:")
        self.output_label.grid(row=2, column=0, sticky='e', **padding)

        self.output_dir_entry = tk.Entry(self.master, textvariable=self.output_dir_path, width=40, state='readonly')
        self.output_dir_entry.grid(row=2, column=1, **padding)

        self.output_dir_button = tk.Button(self.master, text="Browse", command=self.browse_output_directory)
        self.output_dir_button.grid(row=2, column=2, **padding)

        # Output Filename
        self.output_filename_label = tk.Label(self.master, text="Output Filename:")
        self.output_filename_label.grid(row=3, column=0, sticky='e', **padding)

        self.output_filename_entry = tk.Entry(self.master, textvariable=self.output_filename, width=40)
        self.output_filename_entry.grid(row=3, column=1, **padding)

        # Sync and Merge Button
        self.sync_button = tk.Button(self.master, text="Sync and Merge", command=self.start_sync_merge, bg="blue", fg="white")
        self.sync_button.grid(row=4, column=1, **padding)

        # Progress Bar
        self.progress = ttk.Progressbar(self.master, orient='horizontal', mode='determinate', length=500)
        self.progress.grid(row=5, column=0, columnspan=3, **padding)

        # Status Label
        self.status_label = tk.Label(self.master, text="Status: Idle")
        self.status_label.grid(row=6, column=0, columnspan=3, **padding)

        # Encoder Label
        self.encoder_label = tk.Label(self.master, text="Encoder: Not determined yet")
        self.encoder_label.grid(row=7, column=0, columnspan=3, **padding)

        # Disclaimer Label
        self.disclaimer_label = tk.Label(self.master, text="", fg='red')
        self.disclaimer_label.grid(row=8, column=0, columnspan=3, **padding)

    def browse_main_video(self):
        file_path = filedialog.askopenfilename(title="Select Main Video", filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")])
        if file_path:
            self.main_video_path.set(file_path)

    def browse_reaction_video(self):
        file_path = filedialog.askopenfilename(title="Select Reaction Video", filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")])
        if file_path:
            self.reaction_video_path.set(file_path)

    def browse_output_directory(self):
        directory_path = filedialog.askdirectory(title="Select Output Directory")
        if directory_path:
            self.output_dir_path.set(directory_path)

    def start_sync_merge(self):
        if not self.main_video_path.get():
            messagebox.showerror("Error", "Please select the main video.")
            return
        if not self.reaction_video_path.get():
            messagebox.showerror("Error", "Please select the reaction video.")
            return
        if not self.output_dir_path.get():
            messagebox.showerror("Error", "Please select the output directory.")
            return
        if not self.output_filename.get().strip():
            messagebox.showerror("Error", "Please enter a valid output filename.")
            return

        # Ensure output filename ends with .mp4
        if not self.output_filename.get().lower().endswith('.mp4'):
            overwrite_extension = messagebox.askyesno("Warning", "Output filename does not end with .mp4. Append the extension automatically?")
            if overwrite_extension:
                self.output_filename.set(self.output_filename.get() + ".mp4")
            else:
                return  # User chose not to append, exit the function

        # Construct the full output path
        output_full_path = os.path.join(self.output_dir_path.get(), self.output_filename.get())

        # Check if output file already exists
        if os.path.isfile(output_full_path):
            overwrite = messagebox.askyesno("Overwrite Confirmation", f"The file '{self.output_filename.get()}' already exists in the selected directory. Do you want to overwrite it?")
            if not overwrite:
                return

        # Disable buttons to prevent multiple operations
        self.sync_button.config(state='disabled')
        self.main_video_button.config(state='disabled')
        self.reaction_video_button.config(state='disabled')
        self.output_dir_button.config(state='disabled')

        # Reset progress bar and status
        self.progress['value'] = 0
        self.status_label.config(text="Status: Processing...")
        self.encoder_label.config(text="Encoder: Determining...")
        self.disclaimer_label.config(text="")  # Hide disclaimer initially

        # Start the processing in a separate thread
        threading.Thread(target=self.sync_merge_process, args=(output_full_path,)).start()

    def sync_merge_process(self, output_full_path):
        try:
            # Define temporary synced reaction video path
            synced_reaction_path = os.path.join(self.output_dir_path.get(), "reaction_video_synced_temp.mp4")

            # Synchronize Reaction Video
            self.update_status("Status: Synchronizing reaction video...")
            synchronize_videos(
                reaction_video_path=self.reaction_video_path.get(),
                synced_reaction_path=synced_reaction_path,
                num_baseline_frames=10,
                threshold=30,
                match_ratio=0.5,
                progress_callback=lambda x: None  # No progress updates for synchronization
            )

            # Start merging step
            self.update_status("Status: Merging videos...")
            self.update_progress(0)

            merge_videos_ffmpeg(
                main_video_path=self.main_video_path.get(),
                synced_reaction_path=synced_reaction_path,
                output_path=output_full_path,
                progress_callback=self.update_progress,
                encoder_callback=self.update_encoder_info
            )

            # After merging
            # Delete the Synchronized Reaction Video
            delete_file(synced_reaction_path)

            # Ensure progress is set to 100%
            self.update_progress(100)

            # Update status
            self.update_status("Status: Completed Successfully!")

            # Show success message and close the application
            self.master.after(0, lambda: self.show_success_and_close(output_full_path))

        except Exception as e:
            self.update_status(f"Status: Error - {str(e)}")
            self.master.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))

        finally:
            # Re-enable buttons
            self.master.after(0, lambda: [
                self.sync_button.config(state='normal'),
                self.main_video_button.config(state='normal'),
                self.reaction_video_button.config(state='normal'),
                self.output_dir_button.config(state='normal')
            ])

    def show_success_and_close(self, output_full_path):
        messagebox.showinfo("Success", f"Videos synchronized and merged successfully.\nSaved at: {output_full_path}")
        self.master.destroy()  # Close the application

    def update_progress(self, value):
        # Ensure the value doesn't exceed 100 and is not negative
        value = max(0, min(value, 100))
        self.master.after(0, lambda: self.progress.config(value=value))

    def update_encoder_info(self, encoder_description):
        self.master.after(0, lambda: self.encoder_label.config(text=f"Encoder: {encoder_description}"))
        if 'libx264' in encoder_description.lower():
            self.master.after(0, lambda: self.disclaimer_label.config(text="You do not have an NVIDIA gpu so the encoding will be much slower"))
        else:
            self.master.after(0, lambda: self.disclaimer_label.config(text=""))

    def update_status(self, message):
        self.master.after(0, lambda: self.status_label.config(text=message))

def main():
    root = tk.Tk()
    app = VideoSyncMergerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()