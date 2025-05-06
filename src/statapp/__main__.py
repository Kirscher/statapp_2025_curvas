"""
Main entry point for the statapp CLI application.

This module sets up the command-line interface and registers all available commands.
"""

from rich import print
import sys
from typer import Typer

from statapp.about import app as about
from statapp.prepare_dataset import app as prepare
from statapp.upload_data import app as upload_data
from statapp.empty_data import app as empty_data
from statapp.train import app as train

from dotenv import load_dotenv

app = Typer(
    context_settings={"help_option_names": ["--help", "-h"]},
    no_args_is_help=True  # Show help when no command is provided
)

app.add_typer(about)
app.add_typer(prepare)
app.add_typer(upload_data)
app.add_typer(empty_data)
app.add_typer(train)

def main() -> None:
    """
    Main entry point for the application.

    This function loads environment variables and runs the Typer application.
    """
    load_dotenv()
    app()

if __name__ == "__main__":
    main()
