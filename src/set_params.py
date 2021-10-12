import yaml
from box import Box
import argparse
import os
import sys


# python_path = os.path.join(os.getcwd())
# sys.path.append(python_path)
# os.environ["PYTHONPATH"] = python_path


def set_params(params_path):
    with open(params_path, 'r', encoding='utf-8') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    params = Box(params)
    return params


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--params', dest='params_path', required=False)
    args = args_parser.parse_args()

    if args.params_path:
        try:
            params_path = args_parser.params_path
        except Exception as ex:
            print(ex)
    else:   
        print(f'Не удалось загрузить параметры, используем параметры по умолчанию')
        params_path = '../params.yaml'
        
    params = set_params(params_path)
    print(f"Параметры: \n {params}")
    
