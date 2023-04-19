import os
import sys
import yaml

def get_config():
    yaml_path = 'model_data/gesture.yaml'
    f = open(yaml_path,'r',encoding='utf-8')
    config = yaml.load(f,Loader =yaml.FullLoader)
    f.close()
    return config

if __name__ == "__main__":
    config = get_config()
    print(config)