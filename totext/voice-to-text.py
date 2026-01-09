import subprocess
import threading
import queue
import os
import pty
import sys
import re
from datetime import datetime
import time

def start_whisper_stream():
    proc = subprocess.Popen(
        [
            "../whisper/build/bin/whisper-stream",
            "-m", "../whisper/models/ggml-tiny.en.bin",
            "--step", "500",
            "--length", "5000",
            "--keep", "200",
            "-l", "en",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    return proc

def normalise_transcript(s: str) -> str:
    s = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', s)
    s = s.replace('\r', '')
    s = s.replace('\n', '')
    s = s.replace('[BLANK_AUDIO]', '')

    parts = re.split(r'\s{2,}', s)
    s = parts[-1] if parts else s

    return s.strip()

def reader(proc, out_queue):
    started = False

    for line in proc.stdout:
        line = line.strip()

        if "[Start speaking]" in line:
            started = True
            print("Start Speaking!")
            continue

        if not started:
            continue

        if not line:
            continue

        if line.startswith(("whisper_", "main:", "init:", "--")):
            continue

        normalised_line = normalise_transcript(line)

        if not normalised_line:
            continue

        out_queue.put(normalised_line)

def main():
    q = queue.Queue()
    proc = start_whisper_stream()

    threading.Thread(target=reader, args=(proc, q), daemon=True).start()

    last_text = ""

    while True:
        try:
            first_text = q.get()

            if (first_text == last_text):
                continue

            print("Speech detected! Processing")
            time.sleep(2)

            final_text = first_text
            while not q.empty():
                final_text = q.get()

            lower_text = final_text.lower()
            print("User: ", final_text)

            if "wake up" in lower_text:
                print("Play wake up animation!")
            elif "what's the time" in lower_text:
                print("The time is: ", datetime.now().time())

            last_text = final_text
        except KeyboardInterrupt:
            print("Keyboard interrupted\n")
            proc.terminate()
            exit()

if __name__ == "__main__":
    main()
