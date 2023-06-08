import typer

app = typer.Typer(help="openfractal-client CLI", add_completion=False)


@app.command()
def dummy():
    typer.echo("Hello World")
