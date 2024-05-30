from typing import Annotated

import typer

from polaris.hub.client import PolarisHubClient
from polaris.hub.settings import PolarisHubSettings

app = typer.Typer(
    add_completion=False,
    help="Polaris is a framework for benchmarking methods in drug discovery.",
)


@app.command("login")
def login(
    client_env_file: Annotated[
        str, typer.Option(help="Environment file to overwrite the default environment variables")
    ] = ".env",
    auto_open_browser: Annotated[
        bool, typer.Option(help="Whether to automatically open the link in a browser to retrieve the token")
    ] = True,
    overwrite: Annotated[
        bool, typer.Option(help="Whether to overwrite the access token if you are already logged in")
    ] = False,
):
    """Authenticate to the Polaris Hub.

    This CLI will use the OAuth2 protocol to gain token-based access to the Polaris Hub API.
    """
    with PolarisHubClient(settings=PolarisHubSettings(_env_file=client_env_file)) as client:
        client.login(auto_open_browser=auto_open_browser, overwrite=overwrite)


@app.command(hidden=True)
def secret():
    # NOTE (cwognum): Empty, hidden command to force Typer to not collapse the subcommand.
    # Added because I anticipate we will want to add more subcommands later on. This will keep
    # the API consistent in the meantime. Once there are other subcommands, it can be removed.
    # See also: https://github.com/tiangolo/typer/issues/315
    raise NotImplementedError()


if __name__ == "__main__":
    app()
