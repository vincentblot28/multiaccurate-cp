import yaml
from pydantic import BaseSettings


def store_config(config: BaseSettings, path: str) -> None:
    with open(str(path), "w") as file:
        yaml.dump(config.dict(), file)
