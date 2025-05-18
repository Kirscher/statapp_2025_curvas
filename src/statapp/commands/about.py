"""
About module for the statapp application.

This module provides information about the project.
"""

import typer
from rich.text import Text

from statapp.utils import utils

app = typer.Typer()

text = Text.assemble(
    ("Medical segmentation: handling inter-rater uncertainty and variability\n", "bold"),
    ("Made with nnU-Net, a framework for out-of-the box image segmentation.\n", "grey53"),
    ("\n", ""),
    ("CLI by ", ""), ("Rémy SIAHAAN--GENSOLLEN", "italic"),
    justify="center")

@app.command()
def about() -> None:
    """
    Display information about the project.

    This command shows details about the project's purpose and contributors.
    """
    utils.info(text)