from rich.text import Text
import typer
import statapp.utils as utils

app = typer.Typer()

text = Text.assemble(
    ("Medical segmentation : handling inter-rater uncertainty and variability\n", "bold"),
    ("Made with nnU-Net, a framework for out-of-the box image segmentation.\n", "grey53"),
    ("\n", ""),
    ("CLI by ", ""), ("Rémy SIAHAAN--GENSOLLEN", "italic"),
    justify="center")

@app.command()
def about():
    utils.info(text)