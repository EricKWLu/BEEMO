import subprocess
import threading
import queue
import os
import pty
import sys

def start_whisper_stream():
    master, slave = pty.openpty()  # pseudo-terminal used to avoid buffering issue creating new lines in terminal

    proc = subprocess.Popen(
        [
            "../whisper/build/bin/whisper-stream",
            "-m", "../whisper/models/ggml-tiny.en.bin",
            "--step", "500",
            "--length", "5000",
            "--keep", "200",
            "-l", "en"
        ],
        stdout=slave,
        stderr=slave,
        text=True,
        bufsize=1,
        close_fds=True
    )

    os.close(slave)  # we only read from master
    return proc, master

def reader(master_fd, out_queue):
    import io
    f = io.TextIOWrapper(os.fdopen(master_fd, 'rb'), encoding='utf-8', newline='\n')

    started = False

    for line in f:
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

        out_queue.put(line)

def main():
    q = queue.Queue()
    proc, master_fd = start_whisper_stream()

    threading.Thread(target=reader, args=(master_fd, q), daemon=True).start()

    while True:
        try:
            text = q.get()
            print("User:", text)
        except KeyboardInterrupt:
            print("Keyboard interrupted\n")
            exit()

if __name__ == "__main__":
    main()
