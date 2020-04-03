from typing import Dict, Tuple

import hydro_serving_grpc as hs_grpc
import pandas as pd
import yaml
from google.protobuf.json_format import MessageToDict


class AlwaysTrueObj(object):
    def __eq__(self, other):
        return True


AnyDimSize = AlwaysTrueObj()

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from type_conversions import HS_TO_NP_DTYPE, NP_TO_HS_DTYPE, NP_DTYPE_TO_ARG_NAME, STR_TO_HS_DTYPE, HS_DTYPE_TO_STR, NAME_TO_DTYPES
import numpy as np


class ContractViolationException(Exception):
    """
    Exception raised after failed compliance check
    """
    pass


class HSContract:
    """
    HydroServingContract contains useful info about input/output
    tensors and verifies passed dictionary for compliance with contract
    """

    def __str__(self) -> str:
        return f"Input tensors = {self.input_names}; Output tensors = {self.output_names} "

    @classmethod
    def from_df(cls, input_df: pd.DataFrame):
        """
        Suggest contract definition for model contract from dataframe
        :param input_df:
        :return:
        """
        inputs = []
        for name, dtype in zip(input_df.columns, input_df.dtypes):
            inputs.append({"name": name,
                           "dtype": dtype,
                           "shape": (-1, 1)})

        outputs = []

        contract_dict = {"inputs": inputs,
                         "outputs": outputs}
        return cls(contract_dict)

    @classmethod
    def from_proto(cls, model_version_proto):
        """
        Contract HSContract from proto definition of model
        :param model_version_proto:
        :return:
        """
        model_proto_dict = MessageToDict(model_version_proto, preserving_proto_field_name=True)
        proto_contract_dict: Dict = model_proto_dict['contract']['predict']
        contract_dict = {}

        input_tensors = []
        for t in proto_contract_dict['inputs']:
            name = t['name']
            shape = tuple([int(s['size']) for s in t['shape'].get('dim', [])])
            dtype = (HS_TO_NP_DTYPE[STR_TO_HS_DTYPE[t['dtype']]])
            input_tensors.append({"name": name,
                                  "shape": shape,
                                  "dtype": dtype})
        contract_dict['inputs'] = input_tensors

        output_tensors = []
        for t in proto_contract_dict['outputs']:
            name = t['name']
            shape = tuple([int(s['size']) for s in t['shape'].get('dim', [])])
            dtype = (HS_TO_NP_DTYPE[STR_TO_HS_DTYPE[t['dtype']]])
            output_tensors.append({"name": name,
                                   "shape": shape,
                                   "dtype": dtype})
        contract_dict['outputs'] = output_tensors

        return HSContract(contract_dict)

    def dump(self, f):
        """
        Dump current contract into YAML f
        :param f:
        :return:
        """
        contract = self.contract_dict

        def tensor_dict_to_yaml(t_dict):
            dtype = HS_DTYPE_TO_STR[NP_TO_HS_DTYPE[t_dict['dtype']]]
            shape = list(t_dict['shape'])
            if not shape:
                shape = "scalar"
            return t_dict['name'], {"shape": shape, "type": dtype}

        yaml_contract = {"kind": "Model",
                         "payload": ["src/", "requirements.txt"],
                         "runtime": "hydrosphere/serving-runtime-python-3.6:0.1.2-rc0",
                         "install-command": "pip install -r requirements.txt"}
        inputs = {}
        outputs = {}

        for t in contract['inputs']:
            n, d = tensor_dict_to_yaml(t)
            inputs[n] = d
        for t in contract['outputs']:
            n, d = tensor_dict_to_yaml(t)
            outputs[n] = d

        yaml_contract['contract'] = {}
        yaml_contract['contract']['inputs'] = inputs
        yaml_contract['contract']['outputs'] = outputs

        yaml.dump(yaml_contract, f, allow_unicode=True, default_flow_style=None)

    @classmethod
    def load(cls, file):
        """
        Load contract from YAML f
        :param file:
        :return:
        """
        yaml_contract = yaml.load(file, Loader=Loader)

        def yaml_to_tensor_dict(t_name, t_dict):
            if t_dict['shape'] == 'scalar':
                shape = tuple()
            else:
                shape = tuple(t_dict['shape'])

            return {"name": t_name,
                    "dtype": (HS_TO_NP_DTYPE[NAME_TO_DTYPES[t_dict['type']]]),
                    "shape": shape}

        contract_dict = {}
        inputs = []
        outputs = []
        for k, v in yaml_contract['contract']['inputs'].items():
            inputs.append(yaml_to_tensor_dict(k, v))
            pass
        for k, v in yaml_contract['contract']['outputs'].items():
            outputs.append(yaml_to_tensor_dict(k, v))

        contract_dict['inputs'] = inputs
        contract_dict['outputs'] = outputs
        return cls(contract_dict)

    def __init__(self, contract_dict):
        self.contract_dict = contract_dict
        outputs = self.contract_dict['outputs']
        inputs = self.contract_dict['inputs']

        self.output_names = [x['name'] for x in outputs]
        self.output_shapes = dict([(x['name'], x['shape']) for x in outputs])
        self.output_dtypes = dict([(x['name'], x['dtype']) for x in outputs])
        self.number_of_output_tensors = len(outputs)

        self.input_names = [x['name'] for x in inputs]
        self.input_shapes = dict([(x['name'], x['shape']) for x in inputs])
        self.input_is_scalar = dict([(k, v == tuple([])) for k, v in self.input_shapes.items()])
        self.input_dtypes = dict([(x['name'], x['dtype']) for x in inputs])
        self.number_of_input_tensors = len(inputs)

    def verify_input_dict(self, x: Dict[str, np.array]) -> (bool, str):
        """
        False if x is not compliant with this contract
        """
        valid = True
        error_message = ""

        # Check for tensor names
        tensor_names = set(x.keys())
        input_names = set(self.input_names)

        if not tensor_names == input_names:
            valid = False
            tensor_names_error_message = "Invalid input tensor names.\n"
            missing_fields = input_names.difference(tensor_names)
            extra_fields = tensor_names.difference(input_names)
            if missing_fields:
                tensor_names_error_message += f"Missing tensors: {missing_fields}\n "
            if extra_fields:
                tensor_names_error_message += f"Extra tensors: {extra_fields}.\n"
            error_message += tensor_names_error_message

        # Check for tensor shapes and dtypes
        tensor_shapes_error_message = ""
        tensor_dtypes_error_message = ""
        for tensor_name in self.input_names:
            if not HSContract.shape_compliant(self.input_shapes[tensor_name], x[tensor_name].shape):
                valid = False
                tensor_shapes_error_message += f"Tensor \"{tensor_name}\" is {x[tensor_name].shape}," \
                    f" expected {self.input_shapes[tensor_name]}\n"

            if not HSContract.dtype_compliant(self.input_dtypes[tensor_name], x[tensor_name].dtype):
                valid = False
                tensor_dtypes_error_message += f" Tensor \"{tensor_name}\" is {x[tensor_name].dtype}," \
                    f" expected {self.input_dtypes[tensor_name]}"

        error_message += tensor_shapes_error_message
        error_message += tensor_dtypes_error_message

        return valid, error_message

    def make_tensor_shape_proto(self, name):
        return hs_grpc.TensorShapeProto(dim=[hs_grpc.TensorShapeProto.Dim(size=x) for x in self.input_shapes[name]])

    def get_payload_arg_name(self, tensor_name):
        """
        Get name under which we store data in tensor proto
        dependent on dtype
        :param tensor_name:
        :return:
        """
        np_dtype = self.input_dtypes[tensor_name]
        return NP_DTYPE_TO_ARG_NAME[np_dtype]

    def make_proto(self, input_dict: Dict[str, np.array]) -> Dict[str, hs_grpc.TensorProto]:
        """
        Transforms Dict with ndarray tensors into dict with TensorProtos, according to
        contract
        :param input_dict:
        :return:
        """
        input_proto_dict = dict()
        for k, v in input_dict.items():
            kwargs = {self.get_payload_arg_name(k): v.flatten(),
                      "dtype": NP_TO_HS_DTYPE[self.input_dtypes[k]],
                      "tensor_shape": self.make_tensor_shape_proto(k)}
            input_proto_dict[k] = hs_grpc.TensorProto(**kwargs)

        return input_proto_dict

    @staticmethod
    def dtype_compliant(contract_dtype: np.dtype, tensor_dtype: np.dtype) -> bool:
        # Use np.can_cast rules in future?
        return tensor_dtype == contract_dtype

    @staticmethod
    def shape_compliant(contract_shape: Tuple, tensor_shape: Tuple) -> bool:
        if len(contract_shape) == 0:
            # scalar input can be used in following scenarios
            if tensor_shape == tuple():
                return True
            else:
                return max(tensor_shape) == 1  # All dimensions are equal to 1

        if len(contract_shape) == len(tensor_shape):
            possible_contract_shape = tuple([AnyDimSize if s == -1 else s for s in contract_shape])
            return possible_contract_shape == tensor_shape
        else:
            return False

    def decode_request(self, proto):
        """
        Transform imput proto dict into dictionary with np.arrays. We trust gateway to return us only
        valid input, so we do not check contract compliance
        """
        input_dict = {}
        for k, v in MessageToDict(proto, preserving_proto_field_name=True)['inputs'].items():
            dtype = (HS_TO_NP_DTYPE[STR_TO_HS_DTYPE[v['dtype']]])
            value = proto.inputs[k].__getattribute__(NP_DTYPE_TO_ARG_NAME[dtype])
            if v['tensor_shape']:
                shape = [int(s['size']) for s in v['tensor_shape']['dim']]
                value = np.array(value, dtype=dtype).reshape(shape)
            else:
                value = np.array(value, dtype=dtype).item()

            input_dict[k] = value

        return input_dict

    def decode_response(self, proto: hs_grpc.tf.api.predict_pb2.PredictResponse) -> Dict[str, np.array]:
        """
        Transform PredictResponse proto into dictionary with np.arrays. We trust gateway to return us only
        valid output, so we do not check contract compliance
        """
        output_dict = {}
        for k, v in MessageToDict(proto, preserving_proto_field_name=True)['outputs'].items():
            dtype = (HS_TO_NP_DTYPE[STR_TO_HS_DTYPE[v['dtype']]])
            value = proto.outputs[k].__getattribute__(NP_DTYPE_TO_ARG_NAME[dtype])
            if v['tensor_shape']:
                shape = [int(s['size']) for s in v['tensor_shape']['dim']]
                value = np.array(value, dtype=dtype).reshape(shape)
            else:
                value = np.array(value, dtype=dtype).item()

            output_dict[k] = value

        return output_dict

    def __check_for_python_builtins(self, d: Dict) -> Dict:
        """
        Checks for python builtin dtypes in scalar fields of input dict
        and transforms them into suitable versions of np.dtype. If no
        safe conversion is available, raise an Exception
        :param d: input dict
        :return: output dict, without python built-in scalar dtypes
        """
        new_dict = {}
        for k, v in d.items():
            if self.input_is_scalar[k]:
                if v in (0, 1):
                    # Minimum scalar dtype for 0 or 1 is `uint8`, but it
                    # cannot be casted into `bool` safely. So, we detect
                    # for bool scalars by hand.
                    min_input_dtype = np.bool
                else:
                    min_input_dtype = np.min_scalar_type(v)

                if np.can_cast(min_input_dtype, self.input_dtypes[k]):
                    new_dict[k] = self.input_dtypes[k](v)
                else:
                    raise ValueError(f"Can not cast {k} from {min_input_dtype} to {self.input_dtypes[k]}")
            else:
                new_dict[k] = v
        return new_dict

    def make_input_dict(self, x, kwargs) -> Dict[str, np.array]:
        """
        Transform input into input_dict which is used as core representation to check contracts. Input
        must be provided exclusively either as single argument x or by kwargs.
        :param self:
        :param x: Input which will be decoded implicitly into Dict[str, np.array]
        :type x: Union[pd.Series, Dict, pd.DataFrame, np.array]
        :param kwargs: Dictionary with tensors split according to tensor names, as specified in contract
        :type kwargs: Dict[str, np.array]
        :return:
        """
        if len(kwargs) > 0 and x is not None:
            raise ValueError("Too much arguments. Pass your data either in \'x\' or kwargs")

        if x is None and len(kwargs) is None:
            raise ValueError("No data is passed")

        if len(self.input_names) == 1:
            return self.__make_input_dict_from_single_tensor(x, kwargs)

        if type(x) is dict:
            x = self.__check_for_python_builtins(x)
            input_dict = x
        elif type(x) is pd.Series:
            input_dict = dict(x)
        elif type(x) is pd.DataFrame:
            input_dict = dict(x)
            # Reshape into columnar form
            input_dict = dict([(k, np.array(v).reshape((-1, 1))) for k, v in input_dict.items()])
        elif type(x) is np.ndarray:
            raise NotImplementedError("Implicit conversion of single np.array in multi-tensor contracts is not supported")
        elif x is None:
            kwargs = self.__check_for_python_builtins(kwargs)
            input_dict = kwargs
        else:
            raise ValueError(f"Conversion failed. Expected [pandas.DataFrame, numpy.array, pandas.Series, dict], got {type(x)}")
        return input_dict

    def __make_input_dict_from_single_tensor(self, x, kwargs) -> Dict[str, np.array]:
        """
        Similar to `make_input_dict`, but transforms input into input_dict without need of tensor name.
        Tensor name is used implicitly, since contract has only single tensor.
        Tensor name is used implicitly, since contract has only single tensor.
        :param x:
        :param kwargs:
        :return:
        """
        if len(kwargs) > 1:
            raise ValueError(f"Too much arguments. Contract has only singe tensor \'{self.output_names[0]}\'")

        input_name = self.input_names[0]
        if len(kwargs) == 1:
            input_tensor = list(kwargs.values())[0]
        else:
            input_tensor = x

        input_dict = {input_name: input_tensor}
        return input_dict
