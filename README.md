# Lecture2Notes

**Turn a lecture recording into study notes, revision questions and a list of assignments/deadlines, entirely on your own computer.**

Lecture2Notes transcribes a recording with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), then uses a local LLM served by [Ollama](https://ollama.com) (default: `gemma3:4b`) to write structured notes. No cloud service, no API key, no upload of your recordings.

---

## Contents
1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Quick start](#quick-start)
6. [Command-line options](#command-line-options)
7. [Output files](#output-files)
8. [Trying it without audio](#trying-it-without-audio)
9. [Demo guide](#demo-guide)
10. [Configuration (.env)](#configuration-env)
11. [Troubleshooting](#troubleshooting)
12. [Performance expectations](#performance-expectations)
13. [Privacy](#privacy)
14. [Project structure and tests](#project-structure-and-tests)
15. [Known limitations](#known-limitations)

---

## What it does

- Transcribes MP3, WAV, M4A, MP4 and other audio/video files, keeping real segment timestamps.
- Splits the transcript into overlapping chunks and summarizes each one.
- Merges the summaries into one study document: overview, definitions, formulas, algorithms, examples, common mistakes, quick-revision summary and revision questions.
- Finds **assignments, quizzes, exams and deadlines** in the transcript and saves them as validated JSON with the timestamp and evidence, so you can check the original announcement.
- Can watch a folder and process new recordings automatically (optional).
- Never invents content: if a step fails, the notes say so.

## How it works

```
recording
   -> faster-whisper (local)           timestamped segments
   -> chunker (1200 words, 100 overlap)
   -> per-chunk summaries (Ollama)
   -> hierarchical merge               notes.md
   -> per-chunk action-item extraction (JSON, validated, de-duplicated)
                                        action_items.json  (+ section in notes.md)
```

Design rules worth knowing:

- **Grounded prompts.** The model is told to use only what is in the transcript and to flag uncertain text.
- **Deadlines are never guessed.** `deadline_text` keeps the lecturer's wording (for example "next Friday"). `deadline_date` stays `null` unless you pass `--lecture-date` and the date is supported by the wording. Relative phrases are always flagged `needs_verification`.
- **Action items come from the transcript**, not from the summary, and each needs supporting evidence or it is dropped.
- **Failures are visible.** A chunk that cannot be summarized is recorded, the notes get an "INCOMPLETE" banner and the program exits with code 2.
- **Bounded retries.** Ollama requests have timeouts and a limited number of retries; there are no infinite loops.

## Requirements

| Item | Notes |
|---|---|
| Python 3.10+ | 3.11 or 3.12 is the safest choice |
| [Ollama](https://ollama.com/download) | must be running when you process a lecture |
| FFmpeg | strongly recommended; also used as a fallback decoder |
| Disk space | about 3.5 GB for `gemma3:4b`, plus the Whisper model (about 150 MB for `base`) |
| RAM | 8 GB is workable with the default models; more is better |

A GPU is **not** required.

## Setup

Do these steps once. Always run the commands **from inside the project folder**, the one that contains `lecture2notes.py` and the `l2n` folder.

### 1. Open the project folder

Windows PowerShell:
```powershell
cd "C:\path\to\Lecture2Notes"
dir          # you should see lecture2notes.py, l2n, README.md, requirements.txt
```
Linux / macOS:
```bash
cd /path/to/Lecture2Notes
ls
```

### 2. Create and activate a virtual environment

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`, then activate again.

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
You should see `(.venv)` at the start of your prompt.

### 3. Install the Python packages
```
pip install -r requirements.txt
```

### 4. Install FFmpeg

- Windows: `winget install Gyan.FFmpeg` (or `choco install ffmpeg`), then **close and reopen the terminal**
- Ubuntu/Debian: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

Check it with `ffmpeg -version`.

### 5. Install Ollama and download the model

Install Ollama from <https://ollama.com/download>, start the app (on Linux: `ollama serve`), then:
```
ollama pull gemma3:4b
```

### 6. Verify the setup
```
python lecture2notes.py --help
```
The help text should list `--whisper-model`, `--from-transcript` and `--lecture-date`. The first real transcription also downloads the Whisper model, so internet access is needed once.

## Quick start

Process one recording:
```
python lecture2notes.py lecture.mp3
```
The recording can be anywhere on your computer; give its full path, in quotes if it contains spaces:
```powershell
python lecture2notes.py "C:\Users\you\Downloads\lecture.mp4" --whisper-model base --language en
```

Common variations:
```
python lecture2notes.py lecture.mp3 --model gemma3:4b       # choose the Ollama model
python lecture2notes.py lecture.mp3 --whisper-model base    # smaller, faster transcription
python lecture2notes.py lecture.mp3 --language en           # skip language detection
python lecture2notes.py lecture.mp3 --device cpu            # force CPU for Whisper
python lecture2notes.py lecture.mp3 --lecture-date 2026-10-22
python lecture2notes.py --watch ./inbox                     # process new files automatically
python lecture2notes.py --from-transcript path/to/transcript.json   # skip transcription
```

**Tip for long lectures:** a one-hour lecture can take a long time on a CPU. Try a short clip first. To cut 3 minutes starting at the 10-minute mark:
```
ffmpeg -i lecture.mp4 -ss 00:10:00 -t 180 -vn -acodec libmp3lame clip.mp3
```

### Watch mode

`--watch DIR` needs the optional `watchdog` package (already in `requirements.txt`). Existing and new audio files in the folder are processed one at a time after their size stops changing (so half-copied files are not touched). Finished files move to `DIR/processed/`, failed ones to `DIR/failed/`, so nothing is processed twice. Press Ctrl+C to stop.

## Command-line options

| Option | Meaning |
|---|---|
| `recordings` | one or more audio/video files |
| `--watch DIR` | watch a folder for new recordings |
| `--from-transcript JSON` | use an existing `transcript.json` instead of transcribing |
| `--model NAME` | Ollama text model (default `gemma3:4b`) |
| `--whisper-model NAME` | `tiny`, `base`, `small` (default), `medium`, ... |
| `--language CODE` | for example `en`; default is auto-detect |
| `--output-dir DIR` | where results go (default `notes_out`) |
| `--chunk-size N` | words per chunk (default 1200) |
| `--chunk-overlap N` | overlapping words between chunks (default 100) |
| `--device auto\|cpu\|cuda` | Whisper device |
| `--compute-type TYPE` | for example `int8`, `float16` (default: `int8` on CPU) |
| `--lecture-date YYYY-MM-DD` | the real lecture date, enabling calendar dates for deadlines |
| `--verbose` | detailed logging |
| `--help`, `--version` | |

**Exit codes:** `0` complete, `2` finished with warnings (notes are incomplete), `1` failed, `130` interrupted.

## Output files

Each recording gets its own folder; if the name exists, `_2`, `_3`... is used, so earlier results are never overwritten.

```
notes_out/
  lecture_name/
    notes.md              study notes + readable action items (+ warning banner if incomplete)
    transcript.txt        [HH:MM:SS - HH:MM:SS] text
    transcript.json       structured segments with start/end seconds
    action_items.json     validated, de-duplicated assignments and deadlines
    processing.log        events and failures (no transcript text is logged)
    chunk_summaries.json  intermediate per-chunk summaries
    status.json           running / complete / partial / failed / interrupted
```

Example `action_items.json` entry:
```json
{
  "type": "assignment",
  "description": "Submit the lab report on binary search trees",
  "deadline_text": "next Friday",
  "deadline_date": null,
  "source_timestamp": "00:01:20",
  "evidence": "The lecturer says the lab report must be submitted next Friday.",
  "confidence": "high",
  "needs_verification": true,
  "notes": [],
  "additional_mentions": []
}
```

Example section of `notes.md`:
```markdown
## Action Items and Deadlines
- **[Assignment]** Submit the lab report on binary search trees ⚠ verify
  - Deadline as said: "next Friday" · Calendar date: not determined · Heard at 00:01:20 · Confidence: high
  - Evidence: The lecturer says the lab report must be submitted next Friday.
```

Always check deadlines against the recording at the listed timestamp.

## Trying it without audio

A small synthetic transcript (a binary-search-tree concept, an O(h) formula and a lab-report announcement) is included. It skips Whisper but still needs Ollama running:
```
python lecture2notes.py --from-transcript tests/fixtures/sample_transcript.json
```

## Demo guide

1. Use a 2 to 5 minute clip (see the FFmpeg trimming command above), ideally one where you also say an announcement such as "the lab report is due next Friday".
2. Do one full practice run beforehand so the Whisper model is already downloaded.
3. Run: `python lecture2notes.py demo_clip.mp3 --whisper-model base --language en`. The terminal prints each stage as it completes.
4. Open `notes_out/demo_clip/` and show `transcript.txt` (real timestamps), `notes.md` (notes and revision questions) and `action_items.json` (deadline kept in the lecturer's words, `deadline_date` null, evidence and timestamp attached).
5. Show the safety behavior: stop Ollama and run again to see the clear error message, or run `python -m pytest -q`.
6. Fallback if audio misbehaves: the `--from-transcript` command above.

## Configuration (.env)

Copy `.env.example` to `.env` to change defaults. Command-line options override it, and real environment variables override the file.

| Variable | Default |
|---|---|
| `WHISPER_MODEL` | `small` |
| `GEMMA_MODEL` | `gemma3:4b` |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | `1200` / `100` |
| `TEMPERATURE` | `0.2` |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | `auto` / `auto` |
| `OUTPUT_DIR` | `notes_out` |
| `OLLAMA_TIMEOUT` / `OLLAMA_MAX_RETRIES` | `600` seconds / `3` |
| `OLLAMA_NUM_CTX` / `OLLAMA_NUM_PREDICT` | `8192` / `4096` |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `unrecognized arguments: --whisper-model ...` or `No module named 'ollama'` | You ran a different, older `lecture2notes.py` (often one saved in Downloads). `cd` into this project folder and run `python lecture2notes.py --help` to confirm the options. Delete or rename the old script. |
| `can't open file '...lecture2notes.py'` | You are in the wrong folder. `cd` into the folder where `dir` shows `lecture2notes.py` and `l2n` (watch for a nested folder created when unzipping). |
| `Activate.ps1` not recognized | The `.venv` does not exist in this folder. Run `python -m venv .venv` first. |
| Script execution is blocked | `Set-ExecutionPolicy -Scope Process Bypass` |
| `Cannot reach Ollama` | Open the Ollama app (or run `ollama serve`). |
| `Model 'gemma3:4b' is not installed` | `ollama pull gemma3:4b` |
| `open() got an unexpected keyword argument 'metadata_errors'` | PyAV is too old for your faster-whisper. The tool falls back to FFmpeg automatically (install FFmpeg). To fix the cause: `pip install --upgrade --force-reinstall av`. |
| `cublas64_12.dll is not found` | Whisper tried to use a GPU without the CUDA libraries. The tool falls back to CPU automatically; you can also pass `--device cpu`. |
| `FFmpeg ... missing` or decode errors | Install FFmpeg, reopen the terminal, check `ffmpeg -version`. |
| `Could not load Whisper model` | The first run needs internet to download it. Try `--whisper-model base`. |
| Huggingface symlink or `HF_TOKEN` warnings on Windows | Harmless. Hide the first with `$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1`. |
| Request timed out | Raise `OLLAMA_TIMEOUT`, use a smaller `--chunk-size` or a smaller model. |
| Notes show an INCOMPLETE banner | Some chunks failed; read `processing.log`, then re-run (use `--from-transcript` to skip transcription). |
| Many `demo_2`, `demo_3` folders | Each run (even a failed one) creates a new folder; delete old ones freely. |
| Install fails on Python 3.13 | Some packages lag behind new Python versions; use Python 3.11 or 3.12. |

## Performance expectations

Everything runs sequentially to keep memory use predictable. On a CPU-only laptop, transcription speed depends on the Whisper model (`base` is the fastest practical choice, `medium` is much slower) and a 4B LLM generates text slowly, so a one-hour lecture can take longer than the lecture itself. Test with short clips first. These are general guidelines, not measurements from this project.

## Privacy

All processing happens locally. If you point `OLLAMA_HOST` at a non-local machine, lecture text is sent to that machine and the tool prints a warning. Use only servers you trust, prefer HTTPS and network access controls, and never expose Ollama directly to the internet. Logs do not contain transcript text. Keep `.env` and recordings out of version control (`.gitignore` already excludes them).

## Project structure and tests

```
Lecture2Notes/
  lecture2notes.py        entry point
  l2n/                    chunking, transcription, Ollama client, summarizing,
                          action items, outputs, watcher, pipeline, CLI
  tests/                  pytest suite and a sample transcript fixture
  requirements.txt        runtime dependencies
  requirements-dev.txt    adds pytest
  .env.example  .gitignore  inbox/  notes_out/
```

Run the tests (no models needed; Whisper and Ollama are mocked or faked):
```
pip install -r requirements-dev.txt
python -m pytest -q
```
`tests/test_cli_fake_server.py` runs the real HTTP client and CLI against a fake local Ollama server. Passing tests verify the logic and wiring, not the quality of notes from real audio or a real model.

## Known limitations

- Quality depends on audio quality and on the small default models; a 4B model can miss or misread announcements.
- Speaker separation and mixed-language lectures are not handled.
- Speech-recognition errors are flagged where possible, not corrected.
- There is no automatic mid-run resume; use `--from-transcript` to avoid repeating transcription.
- The FFmpeg fallback decoder loads the whole waveform into memory (about 230 MB per hour of audio).
- To improve deadline accuracy: provide `--lecture-date`, use a larger Whisper model and a larger Ollama model, and always verify against the recording.
