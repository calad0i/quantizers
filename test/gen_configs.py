"""The C++ reference quantizers of test_quantizer_match.py: configs, ctypes symbols, and the sources compiled by cpp_source/Makefile."""

from collections import defaultdict
from itertools import product
from pathlib import Path

CPP_SOURCE = Path(__file__).parent / 'cpp_source'
BUILD = CPP_SOURCE / 'build'
LIB_PATH = BUILD / 'libqtest.so'

MATCH_TEST_S = [0, 1]
MATCH_TEST_W = [1, 2, 7, 12]
MATCH_TEST_I = [-5, 0, 8]
MATCH_TEST_RND = ['TRN', 'TRN_ZERO', 'RND', 'RND_CONV', 'RND_ZERO', 'RND_INF', 'RND_MIN_INF']
MATCH_TEST_SAT = ['WRAP', 'SAT', 'SAT_SYM']

# (S, W, I, RND, SAT). Signed 1-bit SAT_SYM is excluded as it is not symmetric at all...
MATCH_CONFIGS_FIXED = [
    (S, W, I, RND, SAT)
    for S, W, I, RND, SAT in product(MATCH_TEST_S, MATCH_TEST_W, MATCH_TEST_I, MATCH_TEST_RND, MATCH_TEST_SAT)
    if not (S and W == 1 and SAT == 'SAT_SYM')
]

# (M, E, E0), E0=1 for the IEEE bias
MATCH_CONFIGS_FLOAT = list(product([2, 6], [2, 6], [0, 1, 3, 5]))


def _id(*conf) -> str:
    return '_'.join(str(v).replace('-', 'm') for v in conf)  # no '-' in C identifiers


def fixed_symbol(S: int, W: int, I: int, RND: str, SAT: str) -> str:
    return f'fixed_{_id(S, W, I, RND, SAT)}'


def float_symbol(M: int, E: int, E0: int) -> str:
    return f'float_{_id(M, E, E0)}'


def _c_fn(symbol: str, fn: str) -> str:
    return f'extern "C" void {symbol}(const float *in, float *out, size_t n) {{ qtest::{fn}(in, out, n); }}'


def _fixed_fn(S: int, W: int, I: int, RND: str, SAT: str) -> str:
    return f'quantize<ac_fixed<{W}, {I}, {"true" if S else "false"}, AC_{RND}, AC_{SAT}>>'


def gen_configs():
    """Write one source per group of configs, as parsing the headers dominates the compile time. Unchanged sources are not
    rewritten, so make only rebuilds what changed."""
    groups: dict[str, list[str]] = defaultdict(list)
    for conf in MATCH_CONFIGS_FIXED:
        groups[f'fixed_{_id(*conf[:3])}'].append(_c_fn(fixed_symbol(*conf), _fixed_fn(*conf)))
    for M, E, E0 in MATCH_CONFIGS_FLOAT:
        groups['float'].append(_c_fn(float_symbol(M, E, E0), f'minifloat<{M}, {E}, {E0}>'))
    groups['binary'].append(_c_fn('binary', 'binary'))
    groups['binary'].append(_c_fn('ternary', _fixed_fn(1, 2, 2, 'RND_CONV', 'SAT_SYM')))

    BUILD.mkdir(exist_ok=True)
    for path in BUILD.glob('*.cpp'):
        if path.stem not in groups:
            path.unlink()  # dropped config group
    for name, fns in groups.items():
        src = '#include "quantizers.h"\n\n' + '\n'.join(fns) + '\n'
        path = BUILD / f'{name}.cpp'
        if not path.exists() or path.read_text() != src:
            path.write_text(src)
