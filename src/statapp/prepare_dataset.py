from rich.text import Text
import typer
import statapp.utils as utils

app = typer.Typer()

text = Text.assemble(("TODO", "bold red"))

@app.command()
def prepare():
    utils.info(text)