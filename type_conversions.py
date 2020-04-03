import numpy as np

from hydro_serving_grpc import DT_STRING, DT_BOOL, \
    DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, \
    DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, \
    DT_UINT64, DT_COMPLEX64, DT_COMPLEX128

NP_TO_HS_DTYPE = {
    np.dtype(np.int8): DT_INT8,
    np.dtype(np.int16): DT_INT16,
    np.dtype(np.int32): DT_INT32,
    np.dtype(np.int64): DT_INT64,
    np.dtype(np.uint8): DT_UINT8,
    np.dtype(np.uint16): DT_UINT16,
    np.dtype(np.uint32): DT_UINT32,
    np.dtype(np.uint64): DT_UINT64,
    np.dtype(np.float16): DT_HALF,
    np.dtype(np.float32): DT_FLOAT,
    np.dtype(np.float64): DT_DOUBLE,
    np.dtype(np.float128): None,
    np.dtype(np.complex64): DT_COMPLEX64,
    np.dtype(np.complex128): DT_COMPLEX128,
    np.dtype(np.complex256): None,
    np.dtype(np.bool): DT_BOOL,
    np.dtype(np.object): None,
    np.dtype(np.str): DT_STRING,
    np.dtype(np.dtype(np.void)): None
}

HS_TO_NP_DTYPE = dict([(v, k) for k, v in NP_TO_HS_DTYPE.items()])

HS_DTYPE_TO_STR = {
    DT_STRING: "DT_STRING",
    DT_BOOL: "DT_BOOL",

    DT_HALF: "DT_FLOAT16",
    DT_FLOAT: "DT_FLOAT32",
    DT_DOUBLE: "DT_DOUBLE",

    DT_INT8: "DT_INT8",
    DT_INT16: "DT_INT16",
    DT_INT32: "DT_INT32",
    DT_INT64: "DT_INT64",

    DT_UINT8: "DT_UINT8",
    DT_UINT16: "DT_UINT16",
    DT_UINT32: "DT_UINT32",
    DT_UINT64: "DT_UINT64",

    DT_COMPLEX64: "DT_COMPLEX64",
    DT_COMPLEX128: "DT_COMPLEX128"
}

STR_TO_HS_DTYPE = dict([(v, k) for k, v in HS_DTYPE_TO_STR.items()] + [("DT_FLOAT", DT_FLOAT)])

NP_DTYPE_TO_ARG_NAME = {
    np.dtype(np.float16): "half_val",
    np.dtype(np.float32): "float_val",
    np.dtype(np.float64): "double_val",

    np.dtype(np.int8): "int_val",
    np.dtype(np.int16): "int_val",
    np.dtype(np.int32): "int_val",
    np.dtype(np.int64): "int64_val",
    np.dtype(np.uint8): "int_val",
    np.dtype(np.uint16): "int_val",
    np.dtype(np.uint32): "uint32_val",
    np.dtype(np.uint64): "uint64_val",
    np.dtype(np.float128): None,
    np.dtype(np.complex64): "scomplex_val",
    np.dtype(np.complex128): "dcomplex_val",
    np.dtype(np.complex256): None,
    np.dtype(np.bool): "bool_val",
    np.dtype(np.object): None,
    np.dtype(np.str): "string_val",
    np.dtype(np.void): None

}

NAME_TO_DTYPES = {
    "string": DT_STRING,
    "bool": DT_BOOL,

    "float16": DT_HALF,
    "half": DT_HALF,
    "float32": DT_FLOAT,
    "float64": DT_DOUBLE,
    "double": DT_DOUBLE,

    "int8": DT_INT8,
    "int16": DT_INT16,
    "int32": DT_INT32,
    "int64": DT_INT64,

    "uint8": DT_UINT8,
    "uint16": DT_UINT16,
    "uint32": DT_UINT32,
    "uint64": DT_UINT64,

    "complex64": DT_COMPLEX64,
    "complex128": DT_COMPLEX128,
}

DTYPE_TO_NAMES = {
    DT_STRING: "string",
    DT_BOOL: "bool",

    DT_HALF: "float16",
    DT_FLOAT: "float32",
    DT_DOUBLE: "float64",

    DT_INT8: "int8",
    DT_INT16: "int16",
    DT_INT32: "int32",
    DT_INT64: "int64",

    DT_UINT8: "uint8",
    DT_UINT16: "uint16",
    DT_UINT32: "uint32",
    DT_UINT64: "uint64",

    DT_COMPLEX64: "complex64",
    DT_COMPLEX128: "complex128"
}
