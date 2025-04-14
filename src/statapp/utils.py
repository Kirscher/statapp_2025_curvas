from rich import print
from rich.panel import Panel
from rich.text import Text

name = "ENSAE 2025 applied statistics project"
version = "Version 0.1.0"

def info(text: Text) -> None:
    panel = Panel(text, title=name, title_align="left", subtitle=version, subtitle_align="left")
    print(panel)
    return None

def pretty_print(text: str) -> None:
    print(text)
    return None