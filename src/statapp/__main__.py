"""
Main entry point for the statapp CLI application.

This module sets up the command-line interface and registers all available commands.
"""

from rich import print
import sys
from typer import Typer

from statapp.commands.about import app as about
from statapp.commands.prepare_dataset_nnunet import app as prepare_nnunet
from statapp.commands.prepare_dataset import app as prepare
from statapp.commands.upload_data import app as upload_data
from statapp.commands.upload_artifacts import app as upload_artifacts
from statapp.commands.empty_data import app as empty_data
from statapp.commands.train import app as train

from dotenv import load_dotenv

app = Typer(
    context_settings={"help_option_names": ["--help", "-h"]},
    no_args_is_help=True  # Show help when no command is provided
)

app.add_typer(about)
app.add_typer(prepare_nnunet)
app.add_typer(prepare)
app.add_typer(upload_data)
app.add_typer(upload_artifacts)
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
