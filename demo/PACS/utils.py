from enum import Enum

import numpy as np
import pandas as pd
from hydro_serving_grpc.reqstore import reqstore_client

from contract import HSContract


def get_reqstore_request(model_contract, rclient: reqstore_client.ReqstoreClient, folder, timestamp, uid: int):
    request_response: reqstore_client.TsRecord = rclient.get(folder, timestamp, uid)
    request = request_response.entries[0].request
    input_tensors = model_contract.decode_request(request)
    return input_tensors


def extract_subsample_from_columnar_reqstore_entries(rs_entries, model) -> pd.DataFrame:
    requests = [model.contract.decode_request(e.request) for e in rs_entries]

    # concat requests into dataframe format
    rs = []
    feature_order = model.contract.input_names
    for r in requests:
        column_arrays = []
        for feature_name in feature_order:
            column_arrays.append(r[feature_name])
        df = pd.DataFrame(np.hstack(column_arrays), columns=feature_order)
        rs.append(df)
    reqstore_data = pd.concat(rs)

    # Sort columns according to feature names order
    reqstore_data = reqstore_data.loc[:, model.contract.input_names]
    reqstore_data.drop_duplicates(inplace=True)
    return reqstore_data


def extract_subsample_from_scalar_reqstore_entries(rs_entries, model) -> pd.DataFrame:
    requests = [model.contract.decode_request(e.request) for e in rs_entries]

    # concat requests into dataframe format
    rs = []
    feature_order = model.contract.input_names
    for r in requests:
        column_arrays = []
        for feature_name in feature_order:
            column_arrays.append([r[feature_name]])
        df = pd.DataFrame(np.hstack(column_arrays).reshape((1, -1)), columns=feature_order)
        rs.append(df)
    reqstore_data = pd.concat(rs)

    # Sort columns according to feature names order
    reqstore_data = reqstore_data.loc[:, model.contract.input_names]
    reqstore_data.drop_duplicates(inplace=True)
    return reqstore_data


def extract_subsample_from_tensor_reqstore_entries(rs_entries, model) -> pd.DataFrame:
    requests = [model.contract.decode_request(e.request)['input'] for e in rs_entries]
    data = np.vstack(requests)
    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)
    return df


def fetch_reqstore_entries(rs_client, model, subsample_size):
    # Get subsample to work with
    model_id = model.id
    rs_entries = []
    for r in rs_client.subsampling(str(model_id), amount=subsample_size):
        rs_entries.extend(r.entries)

    valid_rs_entries = list(filter(lambda x: x.binary != b"", rs_entries))
    return valid_rs_entries


class AlwaysTrueObj(object):
    def __eq__(self, other):
        return True


AnyDimSize = AlwaysTrueObj()


def contract_is_supported_by_rise(contract):
    # TODO check for tensor profiling type

    if 'probabilities' not in contract.output_names:
        return False

    rise_supported_input_shapes = [(-1, AnyDimSize, AnyDimSize, 3),
                                   (-1, AnyDimSize, AnyDimSize, 1),
                                   (-1, AnyDimSize, AnyDimSize),
                                   (AnyDimSize, AnyDimSize, 3),
                                   (AnyDimSize, AnyDimSize, 1),
                                   (AnyDimSize, AnyDimSize)]

    if contract.number_of_input_tensors != 1:
        return False
    else:
        input_tensor_name = contract.input_names[0]
        input_tensor_shape = contract.input_shapes[input_tensor_name]
        if input_tensor_shape not in rise_supported_input_shapes:
            return False

        # Accept only floats, integers, unsigned integers
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html#numpy.dtype.kind
        input_tensor_dtype = contract.input_dtypes[input_tensor_name]
        if input_tensor_dtype.kind not in ('i', 'f', 'u'):
            return False

    return True


class TabularContractType(Enum):
    INVALID = -1
    SCALAR = 0
    COLUMNAR = 1
    SINGLE_TENSOR = 2


def get_tabular_contract_type(contract: HSContract) -> TabularContractType:
    tensor_shapes = list(contract.input_shapes.values())

    if len(tensor_shapes) == 1 and 0 < len(tensor_shapes[0]) < 3:
        return TabularContractType.SINGLE_TENSOR
    elif all([shape == tuple() for shape in tensor_shapes]):
        return TabularContractType.SCALAR
    elif all([shape == (-1, AnyDimSize) for shape in tensor_shapes]):
        return TabularContractType.COLUMNAR
    else:
        return TabularContractType.INVALID


def contract_is_supported_by_anchor(contract):
    # TODO check for tensor profiling type (?)

    if 'classes' not in contract.output_names:
        return False

    if contract.number_of_input_tensors == 1:
        single_tensor_input_name = contract.input_names[0]
        if contract.input_names[0] != "input":
            return False
        single_tensor_input_shape = contract.input_shapes[single_tensor_input_name]
        if single_tensor_input_shape not in [(-1, AnyDimSize)]:
            return False
    else:
        tabular_contract_type = get_tabular_contract_type(contract)

        if tabular_contract_type == TabularContractType.INVALID:
            return False

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html#numpy.dtype.kind
    tensor_dtypes = list(contract.input_dtypes.values())
    if any([dt.kind not in ("i", "u", "f") for dt in tensor_dtypes]):
        return False

    return True


def get_supported_endpoints(contract):
    supported_endpoints = []

    if contract_is_supported_by_rise(contract):
        supported_endpoints.append("rise")
    if contract_is_supported_by_anchor(contract):
        supported_endpoints.append("anchor")

    return supported_endpoints
