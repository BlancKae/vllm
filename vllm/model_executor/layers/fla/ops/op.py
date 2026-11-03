# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

from vllm.triton_utils import tl, tldevice, triton

if (os.environ.get('FLA_USE_FAST_OPS', '0') == '1'
        and all(hasattr(tldevice, name) for name in
                ('fast_dividef', 'fast_expf', 'fast_logf', 'fast_log2f'))):
    div = tldevice.fast_dividef
    exp = tldevice.fast_expf
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:

    @triton.jit
    def div_normal(x, y):
        return x / y

    div = div_normal
    # Triton 3.4 moved these helpers under ``tl.math`` on some builds.
    # Keep the import path usable during vLLM model registry inspection too.
    _tl_math = getattr(tl, "math", tl)
    exp = getattr(tl, "exp", None) or getattr(_tl_math, "exp", None)
    log = getattr(tl, "log", None) or getattr(_tl_math, "log", None)
    log2 = getattr(tl, "log2", None) or getattr(_tl_math, "log2", None)


if not hasattr(tl, 'gather'):

    @triton.jit
    def gather(src, index, axis, _builder=None):
        # This is a fallback implementation when tl.gather is not supported
        # In order to pass triton compiler, there is no actual gather operation
        return src
else:
    gather = tl.gather
