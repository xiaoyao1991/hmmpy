import itertools
import sys
import math

LABEL_INT_MAP = {
    'institution': 0,
    'tech': 1,
    'title': 2,
    'note': 3,
    'author': 4,
    'location': 5,
    'booktitle': 6,
    'editor': 7,
    'date': 8,
    'pages': 9,
    'journal': 10,
    'publisher': 11,
    'volume': 12,
}

INT_LABEL_MAP = [
    'institution' ,
    'tech' ,
    'title' ,
    'note' ,
    'author' ,
    'location' ,
    'booktitle' ,
    'editor' ,
    'date' ,
    'pages' ,
    'journal' ,
    'publisher' ,
    'volume' ,
]

LABEL_INT_MAP_SIX = {
    'FN': 0,
    'LN': 1,
    'DL': 2,
    'TI': 3,
    'VN': 4,
    'YR': 5,
}

INT_LABEL_MAP_SIX = [
    'FN',
    'LN',
    'DL',
    'TI',
    'VN',
    'YR',
]

def deprecated(fun):
    def wrapper():
        print 'This method is deprecated!!!!!!!!!!!!!!!! '
        fun()
    return wrapper

def to_label(ind):
    if ind == 0:
        return 'FN'
    elif ind == 1:
        return 'LN'
    elif ind == 2:
        return 'DL'
    elif ind == 3:
        return 'TI'
    elif ind == 4:
        return 'VN'
    elif ind == 5:
        return 'YR'
    else:
        return ind

def normalize_length(length):
    if length == 0 or length == 1:
        return length
    else:
        return float(length-1) / length

def vec_to_int(vector):
    return int(''.join([str(x) for x in vector]), 2)

def get_binary_vector(n, bits):
    retval = []
    for i in xrange(0, n):
        tmp_ret = []
        tmp = bin(i)[2:].zfill(bits)
        for c in tmp:
            tmp_ret.append(int(c))
        retval.append(tmp_ret)
    return retval

def log_err(msg):
    sys.stderr.write(msg + '\n')

    
# Computes log(1+x) without losing precision for small values of x.
def log_1p(x):
    if x <= -1.0:
        return float('-inf')    # means something is wrong
    if abs(x) > 0.0001:
        return math.log(1.0 + x)
    return (-0.5 * x + 1.0) * x

# Computes x + y without losing precision using ln(x) and ln(y)
def log_sum(ln1, ln2):
    if ln1 == float('-inf'):
        return ln2
    if ln2 == float('-inf'):
        return ln1

    if ln1 > ln2:
        return ln1 + log_1p(math.exp(ln2 - ln1))
    return ln2 + log_1p(math.exp(ln1 - ln2))

if __name__ == '__main__':
    print get_binary_vector(64, 6)
