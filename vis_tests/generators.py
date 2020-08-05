import os
import random
import string
from itertools import product
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import yaml
from hydro_serving_grpc import DT_STRING, DT_BOOL, \
    DT_DOUBLE, DT_INT32, DataType
from hydrosdk.contract import ProfilingType
from hydrosdk.data.types import DTYPE_ALIASES, DTYPE_ALIASES_REVERSE

NUMERICAL_DTYPES = [DT_DOUBLE, DT_INT32]
CATEGORICAL_DTYPES = [DT_STRING, DT_BOOL, DT_DOUBLE, DT_INT32]
CATEGORICAL_PROFILES = [ProfilingType.CATEGORICAL, ProfilingType.NUMERICAL]
NUMERICAL_PROFILES = [ProfilingType.NUMERICAL, ProfilingType.CONTINUOUS, ProfilingType.INTERVAL, ProfilingType.RATIO]
COMPLEX_PROFILES = [ProfilingType.IMAGE, ProfilingType.VIDEO, ProfilingType.AUDIO, ProfilingType.TEXT]
SCALAR_SHAPE = 'scalar'
COMPLEX_SHAPE = [1, 100]
TEST_MODELS_ROOT = 'test_models'
gen_string = lambda N: ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

SCALAR_DTYPE_ALYAS_TO_PROTO_VAL = {
    "string": 'gen_string(10).encode()',
    "bool": '[random.choice([True, False])]',
    "float64": 'np.random.random(1).flatten().tolist()',
    "int32": '[np.random.randint(0, 100)]'
}

COMPLEX_DTYPE_ALYAS_TO_PROTO_VAL = {
    "string": '[gen_string(10).encode() for _ in range(100)]',
    "bool": '[random.choice([True, False]) for _ in range(100)]',
    "float64": 'np.random.random(100).flatten().tolist()',
    "int32": '[np.random.randint(0, 100) for _ in range(100)]'
}

SCALAR_DTYPE_ALYAS_TO_DATA_GENERATOR = {
    "string": lambda n: [gen_string(10) for _ in range(n)],
    "bool": lambda n: [random.choice([True, False]) for _ in range(n)],
    "float64": lambda n: np.random.random(n).flatten().tolist(),
    "int32": lambda n: [np.random.randint(0, 100) for _ in range(n)]
}

COMPLEX_DTYPE_ALYAS_TO_DATA_GENERATOR = {
    "string": lambda n: [[gen_string(10) for _ in range(100)] for _ in range(n)],
    "bool": lambda n: [[random.choice([True, False]) for _ in range(100)] for _ in range(n)],
    "float64": lambda n: [np.random.random(100).flatten().tolist() for _ in range(n)],
    "int32": lambda n: [[np.random.randint(0, 100) for _ in range(100)] for _ in range(n)]
}


def generate_model_fields(numerical=True, categorical=True, other=True, embeddings=False) -> Dict:
    def __generate_fields__(name: str, dtypes: List[int], profiles: List[ProfilingType],
                            shape: Union[str, Dict] = SCALAR_SHAPE):
        fields = {}
        for i, (dtype, profile) in enumerate(list(product(dtypes, profiles))):
            field_dict = {"shape": shape, "type": DTYPE_ALIASES[dtype], "profile": profile.name}
            fields[f"{name}_{i}"] = field_dict
        return fields

    numerical_fields, categorical_fields, other_fields, embedding_field = {}, {}, {}, {}
    if numerical:
        numerical_fields = __generate_fields__("numerical", NUMERICAL_DTYPES, NUMERICAL_PROFILES)
    if categorical:
        categorical_fields = __generate_fields__("categorical", CATEGORICAL_DTYPES, CATEGORICAL_PROFILES)
    if other:
        no_profiling_fields = __generate_fields__("no_profile", CATEGORICAL_DTYPES, [ProfilingType.NONE])
        complex_shape = COMPLEX_SHAPE
        complex_fields = __generate_fields__("complex", CATEGORICAL_DTYPES, COMPLEX_PROFILES, shape=complex_shape)
        other_fields = {**no_profiling_fields, **complex_fields}
    if embeddings:
        embedding_field = {'embedding': {"shape": COMPLEX_SHAPE, "type": DTYPE_ALIASES[DT_DOUBLE], "profile": ProfilingType.NONE.name}}

    return {**numerical_fields, **categorical_fields, **other_fields, **embedding_field}


def generate_model(name: str, training_data=True, has_embeddings=True, numerical_output=True, categorical_output=True, other_output=True,
                   numerical_input=True, categorical_input=True, other_input=True):
    """
    1. generates inputs/outputs contract
    2. generates training data if needed
    3. generates func_main according to model contract
    4. generates training data if needed
    5. generates demo file to load data to cluster
    :param name:
    :return:
    """
    model_path = f'test_models/{name}'
    os.makedirs(model_path, exist_ok=True)
    input_fields = generate_model_fields(numerical_input, categorical_input, other_input)
    output_fields = generate_model_fields(numerical_output, categorical_output, other_output, embeddings=has_embeddings)
    model_dict = {"kind": "Model", "name": name, "payload": ["src/", "requirements.txt"],
                  "runtime": "hydrosphere/serving-runtime-python-3.6:0.1.2-rc0",
                  "install-command": "pip install -r requirements.txt",
                  "contract": {"name": "predict", "inputs": input_fields,
                               "outputs": output_fields}}
    if training_data:
        training_data = generate_training_data(model_dict, 100)
        training_data.to_csv(os.path.join(model_path, 'training_data.csv'))
        model_dict['training-data'] = 'training_data.csv'

    with open(f'{model_path}/serving.yaml', 'w+') as file:
        file.write(yaml.dump_all([model_dict], default_flow_style=False))
    with open(os.path.join(model_path, 'requirements.txt'), 'w+') as file:
        file.write('numpy==1.16.2')
    func_main_content = generate_func_main(model_dict)
    os.makedirs(os.path.join(model_path, 'src'), exist_ok=True)
    with open(os.path.join(model_path, 'src', 'func_main.py'), 'w+') as file:
        file.write(func_main_content)
    return model_dict


def generate_func_main(model_dict):
    header_str = 'import hydro_serving_grpc as hs\n' \
                 'import numpy as np\n' \
                 'import random\n' \
                 'import string\n\n' \
                 'gen_string = lambda N: \'\'.join(random.choices(string.ascii_uppercase + string.digits, k=N))\n\n'
    outputs = model_dict['contract']['outputs']
    signature_name = model_dict['contract']['name']
    func_def = f"def {signature_name}(**kwargs):\n"

    def generate_proto(name: str, field_dict: Dict):
        field_dtype = field_dict['type']
        field_shape = field_dict['shape']
        if field_shape == SCALAR_SHAPE:
            proto_val = SCALAR_DTYPE_ALYAS_TO_PROTO_VAL[field_dtype]
            tensor_shape = 'hs.TensorShapeProto()'
        elif field_shape == COMPLEX_SHAPE:
            proto_val = COMPLEX_DTYPE_ALYAS_TO_PROTO_VAL[field_dtype]
            tensor_shape = 'hs.TensorShapeProto(dim=[hs.TensorShapeProto.Dim(size=1), hs.TensorShapeProto.Dim(size=100)])'
        return f'\t{name}_proto = hs.TensorProto({field_dtype}_val={proto_val}, dtype=hs.{DataType.Name(DTYPE_ALIASES_REVERSE[field_dtype])}, tensor_shape={tensor_shape})'

    output_proto_definintions = '\n'.join([generate_proto(name, field_dict) for name, field_dict in outputs.items()])
    outputs_response_body = '{' + ', '.join([f'"{name}": {name}_proto' for name in outputs.keys()]) + '}'
    return_predict_response = f'\treturn hs.PredictResponse(outputs={outputs_response_body})'

    func_main_content = header_str + func_def + output_proto_definintions + '\n' + return_predict_response
    return func_main_content


def generate_training_data(model_dict, n_samples=100):
    outputs: Dict = model_dict['contract']['outputs']

    def generate_column(field_dict: Dict):
        field_dtype = field_dict['type']
        if field_dict['shape'] == COMPLEX_SHAPE:
            raise ValueError('Only scalar values are used in csv training data')
        data_generator = SCALAR_DTYPE_ALYAS_TO_DATA_GENERATOR[field_dtype]
        generated_data = data_generator(n_samples)
        print(type(generated_data), len(generated_data))
        return data_generator(n_samples)

    return pd.DataFrame.from_dict({name: generate_column(field_dict) for name, field_dict in outputs.items() if
                                   field_dict['shape'] == SCALAR_SHAPE})
