"""Exception hierarchy. Every expected failure derives from Lecture2NotesError."""


class Lecture2NotesError(Exception):
    """Base class for expected, user-reportable errors."""


class ConfigError(Lecture2NotesError):
    pass


class AudioError(Lecture2NotesError):
    """Audio file missing, empty, unreadable or unsupported."""


class TranscriptionError(Lecture2NotesError):
    """Whisper could not be initialised or produced no usable text."""


class OllamaError(Lecture2NotesError):
    """Base class for Ollama communication problems."""


class OllamaUnavailableError(OllamaError):
    """Ollama server cannot be reached."""


class OllamaModelMissingError(OllamaError):
    """The configured model is not installed in Ollama."""


class OllamaTimeoutError(OllamaError):
    """A request to Ollama timed out."""


class LLMOutputError(OllamaError):
    """The model returned empty or unusable output."""


class OutputError(Lecture2NotesError):
    """Output directory/files cannot be created or written."""
