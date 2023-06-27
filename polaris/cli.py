import typer

app = typer.Typer(add_completion=False)


@app.command()
def dummy():
    typer.echo("Hello World")
