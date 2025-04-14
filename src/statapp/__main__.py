from rich import print
import sys
from typer import Typer

from statapp.about import app as about
from statapp.prepare_dataset import app as prepare

app = Typer(context_settings={"help_option_names": ["--help", "-h"]})

app.add_typer(about)
app.add_typer(prepare)

def main() -> None:
    app()

if __name__ == "__main__":
    main()