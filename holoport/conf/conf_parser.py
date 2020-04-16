import yaml

def parse_conf(path_to_yaml):
    config = None

    try:
        with open(path_to_yaml, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file:', path_to_yaml)

    return config


def main():
    vibe_conf = parse_conf('holoport/conf/vibe_conf_local.yaml')


if __name__ == '__main__':
    main()