import click
from .deploy import deploy_agent

@click.group()
def cli():
    pass

cli.add_command(deploy_agent)

cli()
