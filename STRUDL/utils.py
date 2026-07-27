    
import yaml

def parse_yml(config_path):
    """
    Parses configuration file.

    @return: Configuration file content
    """
    with open(config_path) as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return content
