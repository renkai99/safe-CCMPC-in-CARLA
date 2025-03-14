import os
import json
import pathlib
import math
import numbers
import random
import operator
import collections
import itertools
import functools

import scipy
import numpy as np

_marker = object()

class UtilityException(Exception):
    pass

def hello_world():
    """Test that package installation works"""
    return "Hello World"

####################
# General operations
####################

def classname(x):
    return type(x).__name__

def is_iterable(x):
    try:
        iter(x)
        return True
    except:
        return False

######################
# Useful Basic Classes
######################

class AttrDict(dict):
    """

    Note
    ====
    Do not use copy.copy() or copy.deepcopy() on objects of this class.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    def copy(self):
        return AttrDict(**self)

#################
# File operations
#################

def get_dirname_of(fn):
    """Get absolute path of the immediate directory the file
    or directory is in.

    Parameters
    ==========
    fn : str
        The path e.g. '/path/to/image.png'.

    Returns
    =======
    str
        The dirname e.g. '/path/to'.
    """
    return os.path.dirname(os.path.abspath(fn))

def save_json(filepath, obj):
    """Serialize object as a JSON formatted stream and save to a file.

    Parameters
    ----------
    filepath : str
        Path of file to save JSON.
    obj : dict
        Object to serialize.
    """
    with open(filepath, 'w') as f:
        json.dump(obj, f)

def load_json(filepath):
    """Load JSON file to a Python object.

    Parameters
    ----------
    filepath : str
        Path of file to load JSON.
    """
    with open(filepath) as f:
        return json.load(f)

def path_to_filename(path, with_suffix=True):
    """Get filename from path.
    
    Parameters
    ==========
    path : str
        Path to retrieve file name from e.g. '/path/to/image.png'.
    with_suffix : bool
        Whether to include the suffix of file path in file name.

    Returns
    =======
    str
        The file name of the path e.g. 'image.png'
        or 'image' if `with_suffix` is false.
    """
    p = pathlib.Path(path)
    if with_suffix:
        return str(p.name)
    else:
        return str(p.with_suffix("").name)

def strip_extension(path):
    """Function to strip file extension
    DEPRECATED: use 

    Parameters
    ----------
    path : string
        Absoluate path to a slide

    Returns
    -------
    path : string
        Path to a file without file extension
    """
    p = pathlib.Path(path)
    return str(p.with_suffix(''))

###########################
# Not-functional operations
###########################

def do_on_nested_dict_of_list(f, dl, *args, **kwargs):
    if isinstance(dl, list):
        f(dl, *args, **kwargs)
    elif isinstance(dl, dict):
        for v in dl.values():
            do_on_nested_dict_of_list(f, v, *args, **kwargs)
    else:
        pass

def sort_nested_dict_of_list(dl, **kwargs):
    """Sort lists contained inside nested dict.
    Optional arguments for list.sort() are passed on as arguments."""
    def f(l, **kwargs):
        l.sort(**kwargs)
    do_on_nested_dict_of_list(f, dl, **kwargs)

def shuffle_nested_dict_of_list(dl):
    """Shuffle lists contained inside nested dict."""
    def f(l):
        random.shuffle(l)
    do_on_nested_dict_of_list(f, dl)

def setget_dict_from_dict(d, k):
    try:
        return d[k]
    except KeyError:
        d[k] = { }
        return d[k]

def setget_list_from_dict(d, k):
    try:
        return d[k]
    except KeyError:
        d[k] = [ ]
        return d[k]

#########################
# (Functional) operations
#########################

def identity(x):
    """Just return the input."""
    return x

def range_to_list(*args):
    """Creates a range and converts it to a list."""
    return list(range(*args))

def map_to_list(f, l):
    """Does map operation and then converts map object to a list."""
    return list(map(f, l))

def map_to_ndarray(f, l):
    """Does map operation and then converts map object to a list."""
    return np.array(map_to_list(f, l))

def zip_to_list(*args):
    """Does a zip operation and then converts zip object to a list."""
    return list(zip(*args))

def filter_to_list(f, l):
    """Filter from list elements that return true under f(), returning a list.
    Example: (lambda x: x > 2, [1,2,3,4,5]) -> [3,4,5]"""
    return list(filter(f, l))

def compress_to_list(l, bl):
    """Filter list using a list of selectors, returning a list.
    Example: ([1,2,3,4,5], [True, False, False, True, False]) -> [1,4]"""
    # TODO: DEPRECATED
    return list(itertools.compress(l, bl))

def filter_next(f, l, n=1, default=None):
    """Filter and get next in list."""
    return next(filter(f, l), default)

def filter_to_sublist(f, l, n):
    """Filter and get the first n in list, returning a list.
    Example: (lambda x: x[1] == 'b', zip(range(7),'aabbbcc'), 2) -> [(2, 'b'), (3, 'b')]"""
    return list(itertools.islice(filter(f, l), n))

def subdict(d, ks):
    """Get a sub-dict from a dict containing the set of keys."""
    return {k: d[k] for k in ks if k in d}

def from_dict_with_remainder(d, ks):
    """Get values from dict and return remaining keys. Example:
    ({'a': 0, 'b': 1, 'c': 2, 'd': 3}, 'cdef') -> [2, 3], ['f', 'e']

    Parameters
    ==========
    d : dict
        Dict to search.
    ks : list
        Keys to get values from the dict.

    Returns
    =======
    list
        Values in d with keys in ks.
        Equivalent to the output from
        [d[k] for k in ks if k in d]
    list
        Remaining keys from ks that are not in d.
    """
    leftover = set(ks) - set(d.keys())
    ks = set(ks) & set(d.keys())
    return map_to_list(d.get, ks), list(leftover)

def reduce(*args, **kwargs):
    """

    Parameters
    ==========
    f : (function v, acc: f(v, acc))
    l : iterable
    i : any (optional)
    """
    return functools.reduce(*args, **kwargs)

def count(f, l):
    """Count the number of occurances in a list.
    Parameters
    ==========
    f : function
        Function that maps element in list to bool
    
    l : list
        List to count occurances
    
    Returns
    =======
    int
        Number of occurances
    """
    return len(filter_to_list(f, l))

def flatten_nested_list(arb, include=(), exclude=()):
    """Place all non-list values in a flattened list.
    Values are placed in left-to-right DFS order.
    
    Parameters
    ==========
    arb : list
        Arbitrarily nested list.
    include : tuple
        Tuple of types to include in flattened list.
        Takes precedence over exclude.
    exclude : tuple
        Tuple of types to exclude in flattened list.
    """
    out = []
    if include:
        def f(v):
            if isinstance(v, include):
                out.append(v)
    elif exclude:
        def f(v):
            if not isinstance(v, exclude):
                out.append(v)
    else:
        def f(v):
            out.append(v)
    def _flatten(arb):
        for v in arb:
            if isinstance(v, list):
                _flatten(v)
            else:
                f(v)
    _flatten(arb)
    return out

def select_from_nested_list_at_levelindex(arb, level, index):
    """Select items on index of level in arbitrary nested index.
    Values are placed in left-to-right order.

    Parameters
    ==========
    arb : list
        Arbitrarily nested list.
    level : int
        Level of index to gather items.
    index : int
        Index of list at level of nested list to select items.
    """
    if level == 0:
        return arb
    out = []
    def get_to_level(arb, _level):
        try:
            if 0 < _level:
                for v in arb:
                    if isinstance(v, list):
                        get_to_level(v, _level - 1)
            else:
                out.append(arb[index])
        except (TypeError, IndexError):
            pass
    get_to_level(arb, level)
    return out

def merge_list_of_list(ll):
    """Concatenate iterable of iterables into one list."""
    return list(itertools.chain.from_iterable(ll))

def product_list_of_list(ll):
    """Cartesian product iterable of iterables, returning a list."""
    return list(itertools.product(*ll))

def space_list(l):
    return ' '.join(map(str, l))

def underscore_list(l):
    return space_list(l).replace(' ', '_')

def reverse_list(l):
    return list(reversed(l))

def pairwise(l):
    """Make a list of consecutive pairs given a list. 
    Example: [1,2,3] -> [(1,2),(2,3)]"""
    a, b = itertools.tee(l)
    next(b, None)
    return list(zip(a, b))

def grouper(iterable, n, fillvalue=None):
    """Collect data into non-overlapping fixed-length chunks or blocks.
    Example: ('ABCDEFG', 3, 'x') --> ['ABC' 'DEF' 'Gxx']"""
    args = [iter(iterable)] * n
    return list(itertools.zip_longest(*args, fillvalue=fillvalue))

def pairwise_do(f, l):
    """Make a list by applying operation on consecutive pairs in a list.
    Example: [1,2,3] -> [1+2,2+3]
    """
    a, b = itertools.tee(l)
    next(b, None)
    return [f(i, j) for i, j in zip(a, b)]

def unzip(ll):
    """The inverse operation to zip.
    Example: [('a', 1), ('b', 2), ('c', 3)] -> [('a', 'b', 'c'), (1, 2, 3)]
    """
    return list(zip(*ll))

def deduplicate(l):
    """Remove duplicates from list, keeping the list order."""
    seen = {}
    return [seen.setdefault(x, x) for x in l if x not in seen]

def longest_subsequence(l, cond=None):
    """Get the longest subsequence of the list composed of entries
    where cond is true. For example:
    (lambda l, i: l[i], [1, 0, 1, 1, 1, 0, 1]) -> ((slice(2, 5), 3)
    
    Parameters
    ==========
    l : list or np.array
        The list to get subsequence from. 
    cond : function (l, i) -> bool
        To check whether list entries belong to the subsequence.
        If not passed the check using truthiness. 
    
    Returns
    =======
    slice or None
        The slice object to get the subsequence.
    int
        The length of the subsequence.
        Check that such subsequence exist using the length.
    """
    if not cond:
        cond = lambda x: x
    n = len(l)
    longest_size  = 0
    longest_begin = None
    longest_end   = None
    curr_end = 0
    while curr_end < n:
        if cond(l[curr_end]):
            curr_size = 1
            curr_begin = curr_end
            while curr_end < n - 1 and cond(l[curr_end+1]):
                curr_size += 1
                curr_end  += 1
            if curr_size > longest_size:
                longest_size = curr_size
                longest_begin = curr_begin
                longest_end = curr_end
            curr_end += 1
        curr_end += 1
    if longest_size == 0:
        return None, longest_size
    else:
        return slice(longest_begin, longest_end+1), longest_size

def longest_sequence_using_split(l, split):
    """Get the longest subsequence of the list after it has been
    split into segments. For example if split is the function
    lambda l, i: l[i + 1] != l[i] + 1 if i < len(l) - 1 else False
    then
    ([1, 2, 3, 2, 3, 4, 5, 1, 2], split) -> ((slice(3, 7), 4)
    as split gives
    [1, 2, 3 | 2, 3, 4, 5 | 1, 2]
    
    Parameters
    ==========
    l : list or np.array
        The list to get subsequence.
    split : function (l, i) -> bool
        To check whether to split the list.
    
    Returns
    =======
    slice or None
        The slice object to get the subsequence.
    int
        The length of the subsequence.
    """
    n = len(l)
    longest_size  = 0
    longest_begin = None
    longest_end   = None
    curr_begin = 0
    curr_end   = 0
    while curr_end < n:
        if split(l, curr_end) or curr_end == n - 1:
            if (curr_end - curr_begin + 1)  > longest_size:
                longest_size  = curr_end - curr_begin + 1
                longest_begin = curr_begin
                longest_end   = curr_end
            curr_begin = curr_end + 1
            curr_end   = curr_end + 1
        else:
            curr_end  += 1
    if longest_size == 0:
        return None, longest_size
    else:
        return slice(longest_begin, longest_end+1), longest_size

def longest_consecutive_increasing_subsequence(l):
    """Get the longest consecutively increasing subsequence.
    
    Parameters
    ==========
    list of int
        The list to get subsequence.
    
    Returns
    =======
    slice or None
        The slice object to get the subsequence.
    int
        The length of the subsequence.
    """
    def split(l, i):
        try:
            return l[i + 1] != l[i] + 1
        except:
            return False
    return longest_sequence_using_split(l, split)

def longest_consecutive_decreasing_subsequence(l):
    """Get the longest consecutively decreasing subsequence.
    
    Parameters
    ==========
    list of int
        The list to get subsequence.
    
    Returns
    =======
    slice or None
        The slice object to get the subsequence.
    int
        The length of the subsequence.
    """
    def split(l, i):
        try:
            return l[i + 1] != l[i] - 1
        except:
            return False
    return longest_sequence_using_split(l, split)

def inner_keys_from_nested_dict(d, layers=2):
    """Get the inner keys of a of nested of dict. Example:
    ({'a': {'a1': 1 }, 'b': { 'b1': 1, 'b2': 2, }}, 2) -> [['a', 'b', 'c'], ['a1', 'b1', 'b2']]

    Parameters
    ==========
    d : dict
        Nested dict to get keys from.
    layers : int
        Number of layers of inner keys to extract.
        If layers=1 then equivalent to [list(d.keys())].

    Returns
    =======
    list of list
        Each entry in the list is collection of keys at at that level of the nested dict.
        The collection of keys corresponding to the level may be duplicated.
    """
    ll = []
    vl = [d]
    for _ in range(layers):
        l = []
        f = lambda x: isinstance(x, dict)
        q = collections.deque(filter_to_list(f, vl))
        vl = []
        if not q:
            break
        while q:
            d = q.pop()
            keys, values = unzip(d.items())
            vl.append(values)
            l.append(keys)
        l = merge_list_of_list(l)
        vl = merge_list_of_list(vl)
        ll.append(l)
    return ll

def select(l, i):
    """Select from iterable using an index of a list of indices,
    returning a list with entries in the same order as their
    corresponding indices. Examples:

    """
    if is_iterable(i):
        indices = i
        try:
            return [l[i] for i in indices]
        except TypeError:
            mask = indices_to_selection_mask(indices, max(indices)+1)
            return compress(l, mask)
    else:
        try:
            return l[i]
        except TypeError:
            return next(itertools.islice(l, i, None))

# itertools.* functions should output to list
# Replacements of the *_to_list() functions

def compress(*args, **kwargs):
    """Filter list using a list of selectors, returning a list.
    Example: ([1,2,3,4,5], [True, False, False, True, False]) -> [1,4]"""
    return list(itertools.compress(*args, **kwargs))

def accumulate(*args, **kwargs):
    """Accumulate list, returning a list.
    Example: ([1,2,3] -> [1,3,6]; [1,2,3], initial=100) -> [100,101,103,106]"""
    return list(itertools.accumulate(*args, **kwargs))

def starmap(*args, **kwargs):
    """Star-maps list, returning a list.
    Example: (pow, [(2,5), (3,2), (10,3)]) -> [32, 9, 1000]"""
    return list(itertools.starmap(*args, **kwargs))

def product(*args, **kwargs):
    """Cartesian product of iterables, returning a list.
    Example: ('ABC', range(2)) -> [('A', 0), ('A', 1), ('B', 0), ('B', 1), ('C', 0), ('C', 1)]"""
    return list(itertools.product(*args, **kwargs))

def permutations(*args, **kwargs):
    """Return permutations of elements in a list, returning a list.
    Example: ('ABC', 2) --> [('A', 'B', 'C'), ('A', 'C', 'B'), ..., ('C', 'B', 'A')]"""
    return list(itertools.permutations(*args, **kwargs))

#################################
# more-itertools based operations
#################################

def first(iterable, default=_marker):
    """Return the first item of *iterable*, or *default* if *iterable* is
    empty.

        >>> first([0, 1, 2, 3])
        0
        >>> first([], 'some default')
        'some default'

    If *default* is not provided and there are no items in the iterable,
    raise ``ValueError``.
    """
    try:
        return next(iter(iterable))
    except StopIteration as e:
        if default is _marker:
            raise ValueError(
                'first() was called on an empty iterable, and no '
                'default value was provided.'
            ) from e
        return default

def second(iterable, default=_marker):
    """Return the second item of *iterable*, or *default* if *iterable* is
    empty.

        >>> second([0, 1, 2, 3])
        1
        >>> second([], 'some default')
        'some default'

    If *default* is not provided and there are no second item in the iterable,
    raise ``ValueError``.
    """
    try:
        it = iter(iterable)
        next(it)
        return next(it)
    except StopIteration as e:
        if default is _marker:
            raise ValueError(
                'second() was called on an empty iterable, and no '
                'default value was provided.'
            ) from e
        return default

def divide(n, iterable):
    """Divide the elements from *iterable* into *n* parts, maintaining order.
    Taken from more-itertools with minor modification."""
    if n < 1:
        raise ValueError('n must be at least 1')
    try:
        iterable[:0]
    except TypeError:
        seq = tuple(iterable)
    else:
        seq = iterable

    q, r = divmod(len(seq), n)

    ret = []
    stop = 0
    for i in range(1, n + 1):
        start = stop
        stop += q + 1 if i <= r else q
        ret.append(list(seq[start:stop]))

    return ret

##########################
# Non-numy Math operations
##########################

class Clip(object):
    def __init__(self, low=-1, high=1):
        self.__low = low
        self.__high = high
        
    def __call__(self, x):
        return min(max(self.__low, x), self.__high)

def sgn(x):
    """Get the sign of a number as int. 1.2 -> 1 and -1.2 -> -1"""
    return int(math.copysign(1, x))

def map_01_to_uv(u, v):
    """Get mapping from [0, 1] to [u, v]"""
    return lambda x: x*(v - u) + u

def map_uv_to_01(u, v):
    """Get mapping from [u, v] to [0, 1]"""
    return lambda x: (x - u)/(v - u)

#######################
# Numpy math operations
#######################

## DEPRECATED: use utility.npu

def kronecker_add_vectors(a, b):
    """Kronecker addition of two vectors,
    treating a as a row vector and b as a column vector"""
    return a[None, :] + b[:, None]

def kronecker_mul_vectors(a, b):
    """Kronecker multiplication of two vectors,
    treating a as a row vector and b as a column vector"""
    return np.kron(a[None, :], b[:, None])

def decision_to_value(b, values=(1, -1)):
    """Map binary decisions to corresponding values.
    TODO: DEPRECATED: just use np.where()

    Parameters
    ==========
    b : bool or ndarray of bool
        Decision variables.
    values: tuple
        Values to map decision variables to.
        False -> values[0] and True -> values[1]
        By default False -> 1 and True -> -1.
    
    Returns
    =======
    Number or ndarray of Number
        Values.
    """
    if isinstance(b, bool):
        return values[0]*int(not b) + values[1]*int(b)
    elif isinstance(b, list):
        return map_to_list(lambda x: decision_to_value(x, values=values), b)
    elif isinstance(b, np.ndarray):
        if np.issubdtype(b.dtype, np.dtype('bool')):
            return values[0]*np.logical_not(b) + values[1]*b
        else:
            raise ValueError(f"Cannot handle dtype {str(b.dtype)}")
    else:
        raise NotImplementedError(f"Handing type {type(b)} not implemented.")

def is_positive_semidefinite(X):
    """Check that a matrix is positive semidefinite
    
    Based on:
    https://stackoverflow.com/a/63911811
    """
    if X.shape[0] != X.shape[1]:
        return False
    if not np.all( X - X.T == 0 ):
        return False
    try:
        regularized_X = X + np.eye(X.shape[0]) * 1e-14
        np.linalg.cholesky(regularized_X)
    except np.linalg.linalg.LinAlgError as err:
        if "Matrix is not positive definite"  == str(err):
            return False
        raise err
    return True

def is_positive_definite(X):
    """Check that a matrix is positive definite
    
    Based on:
    https://stackoverflow.com/a/63911811
    """
    if X.shape[0] != X.shape[1]:
        return False
    if not np.all( X - X.T == 0 ):
        return False
    try:
        np.linalg.cholesky(X)
    except np.linalg.linalg.LinAlgError as err:
        if "Matrix is not positive definite"  == str(err):
            return False
        raise err
    return True

def indices_to_selection_mask(indices, n):
    mask = np.full(n, False)
    for idx in indices:
        mask[idx] = True
    return mask

def reflect_radians_about_x_axis(r):
    r = (-r) % (2*np.pi)
    return r

def reflect_radians_about_y_axis(r):
    r = (r + np.pi) % (2*np.pi)
    return r

def determinant_2d(A):
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    return a*d - b*c

def inverse_2d(A):
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    return 1. / (a*d - b*c) * np.array([[d, -b], [-c, a]])

def rotation_2d(theta):
    """2D rotation matrix. If x is a column vector of 2D points then
    `rotation_2d(theta) @ x` gives the rotated points.
    
    Parameters
    ==========
    theta : float
        Rotates points clockwise about origin if theta is positive.
        Rotates points counter-clockwise about origin if theta is negative
    
    Returns
    =======
    np.array
        2D rotation matrix of shape (2, 2).
    """
    return np.array([
            [ np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])

def distances_from_line_2d(points, x_start, y_start, x_end, y_end):
    """Get the distances from each point to a line spanned by line segment from
    (x_start, y_start) to (x_end, y_end). Works for horizontal and vertical lines.
    
    Based on:
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    Parameters
    ==========
    points : np.array or list
        One 2D point, or multiple 2D points of shape (n, 2).
    x_start : float
        Line segment component
    y_start : float
        Line segment component
    x_end : float
        Line segment component
    y_end : float
        Line segment component

    Returns
    =======
    float or np.array
        Distance of point to line, or array of distances from points to line.
    """
    points = np.array(points)
    if points.ndim == 1:
        return np.abs((x_end - x_start)*(y_start - points[1]) - (x_start - points[0])*(y_end - y_start)) \
                / np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    elif points.ndim == 2:
        return np.abs((x_end - x_start)*(y_start - points[:, 1]) - (x_start - points[:, 0])*(y_end - y_start)) \
                / np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    else:
        raise UtilityException(f"Points of dimension {points.ndim} are not 1 or 2")

def vertices_of_bboxes(centers, theta, lw):
    """Get the vertices of N rectanglar bounding boxes given the centers of the boxes,
    the thetas the boxes they are pointing at, and the length and width of the boxes,
    assuming the length and widths of all boxes are the same.

    Parameters
    ==========
    centers : np.ndarray
        The centers of the bounding boxes of shape (N, 2).
    theta : number or np.ndarray
        The direction of the boxes in radians. Theta can be a number specifying the
        direction of all boxes, or a ndarray that specifies the direction for each box
        with shape (N,).
    lw : np.ndarray
        The length and width of the box. It can have shape (2,) and applied to all boxes,
        or have shape (N, 2) to specify the dimensions of each box separately.

    Returns
    =======
    np.ndarray
        The vertices of the boxes of shape (N,4,2).
    """
    lws = np.repeat(lw[None], centers.shape[0], axis=0) if lw.ndim == 1 else lw
    thetas = np.full(centers.shape[0], theta) if np.ndim(theta) == 0 else theta
    C = np.cos(thetas)
    S = np.sin(thetas)
    rot11 = np.stack((-C,  S), axis=-1)
    rot12 = np.stack((-S, -C), axis=-1)
    rot21 = np.stack((-C, -S), axis=-1)
    rot22 = np.stack((-S,  C), axis=-1)
    rot31 = np.stack(( C, -S), axis=-1) 
    rot32 = np.stack(( S,  C), axis=-1)
    rot41 = np.stack(( C,  S), axis=-1)
    rot42 = np.stack(( S, -C), axis=-1)
    # Rot has shape (N, 8, 2)
    Rot = np.stack((rot11, rot12, rot21, rot22, rot31, rot32, rot41, rot42), axis=1)
    # disp has shape (N, 8)
    disp = 0.5 * np.einsum("...jk, ...k ->...j", Rot, lws)
    # centers has shape (N, 8)
    centers = np.tile(centers, (4,))
    return np.reshape(centers + disp, (-1,4,2))

def vertices_from_bbox(center, theta, lw):
    return vertices_of_bboxes(np.array([center]), np.array([theta]), lw)[0]

def interp_and_sample(points, n, interpolation='quadratic'):
    distance = np.cumsum(np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)/distance[-1]
    interpolator =  scipy.interpolate.interp1d(distance, points, kind=interpolation, axis=0)
    return interpolator(np.linspace(0, 1, n))

def place_rectangles_on_intep_curve(points, n, lws, thetas=None, interpolation='quadratic'):
    """Interpolate a curve on points and then place boxes on the curve.
    Calls `scipy.interpolate.interp1d()` to do the interpolate.

    Parameters
    ==========
    points : ndarray
        Points to interpolate of shape (N, 2).
    n : int
        Number of boxes to place on the interpolated curve.
    lws : ndarray
        The length and width of the box. It can have shape (2,) and applied to all boxes,
        or have shape (N, 2) to specify the dimensions of each box separately.
    thetas : number or ndarray (optional)
        The direction of the boxes in radians. Theta can be a number specifying the
        direction of all boxes, or a ndarray that specifies the direction for each box
        with shape (N,). By default the boxes point in the direction of the curve.
    interpolation : str or int (optional)
        Specifies the kind of interpolation for `scipy.interpolate.interp1d()` call.
    
    Returns
    =======
    ndarray
        Vertices of rectangles of shape (n, 4, 2).
    """
    interp_points = interp_and_sample(points, 2*n - 1, interpolation=interpolation)
    if thetas is None:
        X = interp_points[:2*n-2].reshape(-1, 2, 2).astype(complex)
        X = X[:, 1, :] - X[:, 0, :]
        X = X[:, 0] + 1j*X[:, 1]
        thetas = np.angle(X)
        X = interp_points[-2] - interp_points[-1]
        thetas = np.concatenate((thetas, [np.angle(X[0] + 1j*X[1])]))
    centers = interp_points[::2]
    return vertices_of_bboxes(centers, thetas, lws)

def pairs2d_to_halfspace(p1, p2):
    """Get half-space representation dividing the left side and the right side
    of the line formed by two points in R^2. The points x where Ax <= b are on
    the right side of the arrow from p1 to p2.
    
     x_2
      ^
      |  \
      |   \  Ax > b
      |    p1
    --|-----\------> x_1
      |      p2
     Ax <= b  \
      |        \

    Parameters
    ==========
    p1 : ndarray
        First point
    p2 : ndarray
        Second point
    
    Returns
    =======
    np.array
        A where Ax <= b
    int
        b where Ax <= b
    
    """
    p11, p12 = p1
    p21, p22 = p2
    A = np.array([p12-p22, p21-p11])
    b = (p12 - p22)*p11 + (p21 - p11)*p12
    return A, b

def vertices_to_halfspace_representation(vertices):
    """Vertices of convex polytope to half-space representation (A, b).
    where points x, A x <= b are inside the polytope. 
    
    Parameters
    ==========
    vertices : np.array
        Vertices of convex polytope of shape (N, 2) where N is the number of vertices.
        The vertices are sorted in clockwise order along the first axis and N > 2.
        
    Returns
    =======
    np.array
        A where x, Ax <= b are the points of the polytope 
    np.array
        b where x, Ax <= b are the points of the polytope
    """
    vertices = np.concatenate((vertices, vertices[0][None],), axis=0)
    A = []; b = []
    for p1, p2 in pairwise(vertices):
        _A, _b = pairs2d_to_halfspace(p1, p2)
        A.append(_A); b.append(_b)
    A = np.stack(A); b = np.array(b)
    return A, b

#####################################################################
# Sequential reimplementation of some Numpy functions for object type
# Used when types cannot be vectorized
#####################################################################

def obj_matmul(A, B):
    """Non-vectorized multiplication of arrays of object dtype"""
    if len(B.shape) == 1:
        C = np.zeros((A.shape[0]), dtype=object)
        for i in range(A.shape[0]):
            for k in range(A.shape[1]):
                C[i] += A[i,k]*B[k]
    else:
        C = np.zeros((A.shape[0], B.shape[1]), dtype=object)
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                for k in range(A.shape[1]):
                    C[i,j] += A[i,k]*B[k,j]
    return C

def obj_vectorize(f, A):
    if A.ndim == 0:
        return f(A)
    elif A.ndim == 1:
        return np.array([f(a) for a in A])
    else:
        return np.stack([obj_vectorize(f, a) for a in A])

####################
# Dataset operations
####################

def create_sample_pattern(sample_pattern):
    """Given a string of '/' separated words, create a dict of the words and their ordering in the string. Idempotent.

    Parameters
    ----------
    sample_pattern : str or (list of str)
        String of '/' separated words

    Returns
    -------
    dict of str: int
        Empty dict if sample pattern is ''.
        Otherwise each key is a word with value that is the index in the patch ID containing the label corresponding to the word.
    """
    if sample_pattern == '':
        return { }
    elif isinstance(sample_pattern, str):
        sample_pattern = sample_pattern.split('/')
        return {k: i for i,k in enumerate(sample_pattern)}
    else:
        return sample_pattern

class IDMaker(object):
    """Create ID string from ID primitives. For example when constructed via
    
    ```
    carla_id_maker = IDMaker(
        'map_name/episode/agent/frame',
        prefixes={
            'episode':  'ep',
            'agent':    'agent',
            'frame':    'frame'},
        format_spec={
            'episode':  '03d',
            'agent':    '03d',
            'frame':    '08d'})
    ```
    
    Then calling
    
    ```
    carla_id_maker.make_id(map_name='Town04', episode=1, agent=123, frame=1000)
    ```
    
    gives Town04/ep001/agent123/frame00001000
    """

    @staticmethod
    def __clamp(s):
        return "{" + str(s) + "}"

    def make_fstring(self):
        def f(w):
            s = self.__clamp(f"{w}:{self.__format_spec[w]}")
            return self.__prefixes[w] + s
        l = map(f, self.__sample_pattern_lst)
        return '/'.join(l)

    def __init__(self, s, prefixes={}, format_spec={}):
        self.__sample_pattern_str = s
        self.__sample_pattern_lst = s.split('/')
        self.__sample_pattern = create_sample_pattern(s)
        self.__prefixes = prefixes
        self.__format_spec = format_spec
        for w in self.__sample_pattern_lst:
            if w not in self.__prefixes:
                self.__prefixes[w] = ''
            if w not in self.__format_spec:
                self.__format_spec[w] = ''
        self.__fstring = self.make_fstring()

    @property
    def sample_pattern_str(self):
        return self.__sample_pattern_str

    @property
    def sample_pattern(self):
        return self.__sample_pattern
    
    @property
    def fstring(self):
        return self.__fstring
        
    def make_id(self, **kwargs):
        return self.__fstring.format(**kwargs)

    def extract_value(self, ids, label):
        """Extract labels an ID or a list of IDs,
        specifying the word.

        Parameters
        ==========
        id : str
        label : str or (list of str)

        Returns
        =======
        str or (list of str)
        """
        sp = self.__sample_pattern
        if isinstance(ids, str):
            return ids.split('/')[sp[label]]
        else:
            return [id.split('/')[sp[label]] for id in ids]

    def filter_ids(self, ids, filter, inclusive=True):
        """Filter IDs.

        Parameters
        ==========
        ids : list of str
            List of IDs of the same form as given by the string value
            given by `IDMaker.sample_pattern_str`.
            For example if sample_pattern_str is map_name/episode/agent/frame
            then an ID might look like Town02/ep002/agent004/frame00000530.
        filter : dict
            The filter for IDs of form (key, value). For example:
            {'map_name': 'Town02'} will filter for the map_name 'part' of the ID.
        inclusive : bool
            Whether to filter inclusively or exclusively.
            
        Returns
        =======
        list of str
            List of IDs.

        TODO: specify dtype of id ndarray as np.dtype('U')
        """
        for word in filter.keys():
            if not isinstance(filter[word], (list, np.ndarray)):
                filter[word] = [filter[word]]
            
        sp = self.__sample_pattern
        id_nd = np.array([[*id.split('/'), id] for id in ids])
        common_words = set(sp) & set(filter)
        b_nd = np.zeros((len(ids), len(common_words)), dtype=bool)
        for idx, word in enumerate(common_words):
            values = filter[word]
            wd_nd = id_nd[:, sp[word]]
            f = lambda v: wd_nd == v 
            wd_nd_b = map_to_ndarray(f, values)
            b_nd[:, idx] = np.any(wd_nd_b, axis=0)
        if inclusive:
            b_nd = np.all(b_nd, axis=1)
        else:
            b_nd = np.any(b_nd, axis=1)
            b_nd = np.logical_not(b_nd)
        id_nd = id_nd[b_nd, -1]
        return id_nd.tolist()
    
    def group_ids(self, ids, words):
        """Group IDs by in the order of the words in the words array.
        For example if sample_pattern of IDs is
        'annotation/subtype/slide/patch_size/magnification'
        and we have IDs like

        Stroma/MMRd/VOA-1000A/512/20/0_0
        Stroma/MMRd/VOA-1000A/512/10/0_0
        Stroma/MMRd/VOA-1000A/512/20/2_2
        Stroma/MMRd/VOA-1000A/256/10/0_0
        Tumor/POLE/VOA-1000B/256/10/0_0

        Setting words=['patch_size', 'magnification'] gives

        512: {
            20: [
                Stroma/MMRd/VOA-1000A/512/20/0_0
                Stroma/MMRd/VOA-1000A/512/20/2_2
            ],
            10: [
                Stroma/MMRd/VOA-1000A/512/10/0_0
            ]
        },
        256: {
            20: [
            ],
            10: [
                Stroma/MMRd/VOA-1000A/256/10/0_0
                Tumor/POLE/VOA-1000B/256/10/0_0
            ]
        }

        Parameters
        ==========
        ids : iterable of str
            List of sample IDs to group.
        words : list of str
            Words to group IDs by. Order of nested labels correspond to order of words array.
        sample_pattern : dict of (str: int)
            Dictionary describing the structure of the patch ID.
            The words for RL experiments can be 'map', 'episode'.
            The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.
        
        Returns
        =======
        dict
            The grouped IDs.
            Each group is a list.
            The keys are strings.
        dict of str: list
            Labels corresponding to each word in words array.
            The labels are strings.
        """
        sp = self.__sample_pattern
        id_nd = np.array([[*id.split('/'), id] for id in ids], dtype=np.dtype('U'))
        word_to_labels = { }
        for word in words:
            word_to_labels[word] = np.unique(id_nd[:, sp[word]]).tolist()
        def traverse_words(part_id_nd, idx=0):
            if idx >= len(words):
                return part_id_nd[:, -1].tolist()
            else:
                word = words[idx]
                out = { }
                for label in word_to_labels[word]:
                    selector = part_id_nd[:, sp[word]] == label
                    out[label] = traverse_words(
                            part_id_nd[selector, :],
                            idx=idx + 1)
                return out
        return traverse_words(id_nd), word_to_labels

    def group_ids_by_index(self, ids, include=[], exclude=[]):
        """Group IDs by patch pattern words.
        For example if patch_pattern of IDs is
        'annotation/subtype/slide/patch_size/magnification'
        and we have IDs like

        Stroma/MMRd/VOA-1000A/512/20/0_0
        Stroma/MMRd/VOA-1000A/512/10/0_0
        Stroma/MMRd/VOA-1000A/512/20/2_2
        Stroma/MMRd/VOA-1000A/256/20/0_0
        Stroma/MMRd/VOA-1000A/256/10/0_0
        Tumor/POLE/VOA-1000B/256/10/0_0

        Setting include=['patch_size', 'magnification'] gives

        512/10: [
            Stroma/MMRd/VOA-1000A/512/10/0_0
        ],
        512/20: [
            Stroma/MMRd/VOA-1000A/512/20/0_0
            Stroma/MMRd/VOA-1000A/512/20/2_2
        ],
        256/10: [
            Stroma/MMRd/VOA-1000A/256/20/0_0
        ],
        256/20: [
            Stroma/MMRd/VOA-1000A/256/10/0_0
            Tumor/POLE/VOA-1000B/256/10/0_0
        ]

        Setting exclude=['slide', 'magnification'] gives

        Stroma/MMRd/VOA-1000A/0_0: [
            Stroma/MMRd/VOA-1000A/512/20/0_0
            Stroma/MMRd/VOA-1000A/512/10/0_0
            Stroma/MMRd/VOA-1000A/256/20/0_0
            Stroma/MMRd/VOA-1000A/256/10/0_0
        ],
        Stroma/MMRd/VOA-1000A/2_2: [
            Stroma/MMRd/VOA-1000A/512/20/2_2
        ],
        Tumor/POLE/VOA-1000B/0_0: [
            Tumor/POLE/VOA-1000B/256/10/0_0
        ]

        Parameters
        ----------
        patch_ids : list of str
        patch_pattern : dict
            Dictionary describing the directory structure of the patch paths.
            The words are 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'
        include : iterable of str
            The words to group by. By default includes all words.
        
        exclude : iterable of str
            The words to exclude.
        Returns
        -------
        dict of str: list
            The patch IDs grouped by words.
        """
        sp = self.__sample_pattern
        id_nd = np.array([[*id.split('/'), id] for id in ids], dtype=np.dtype('U'))
        words = set(sp) - set(exclude)
        if include:
            words = words & set(include)
        indices = sorted([sp[word] for word in words] + [id_nd.shape[1] - 1])
        id_nd = id_nd[:,indices]
        id_nd = np.apply_along_axis(lambda r: np.array(['/'.join(r[:-1]), r[-1]]), 1, id_nd)
        group = { }
        for common_id, id in id_nd:
            if common_id not in group:
                group[common_id] = []
            group[common_id].append(id)
        return group

"""
TODO: the below is deprecated
"""

def create_sample_id(path, sample_pattern=None, rootpath=None):
    """Create sample ID from path either by
    1) sample_pattern to find the words to use for ID
    2) rootpath to clip the patch path from the left to form patch ID

    Parameters
    ----------
    path : string
        Absolute path to a patch
    sample_pattern : dict of (str: int)
        Dictionary describing the structure of the patch path.
        The words for RL experiments can be 'map', 'episode'.
        The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.
    rootpath : str
        The root directory path containing sample to clip from sample file path.
        Assumes file path contains rootpath.

    Returns
    -------
    str
        Sample ID generated from path.
    """
    if sample_pattern is not None:
        len_of_patch_id = -(len(sample_pattern) + 1)
        patch_id = strip_extension(path).split('/')[len_of_patch_id:]
        return '/'.join(patch_id)
    elif rootpath is not None:
        return strip_extension(path[len(rootpath):].lstrip('/'))
    else:
        return ValueError("Either sample_pattern or rootpath should be set.")

def create_sample_ids(paths, sample_pattern=None, rootpath=None):
    """Apply create_sample_id() for a list of paths.

    Parameters
    ----------
    path : string
        Absolute path to a patch
    sample_pattern : dict of (str: int)
        Dictionary describing the structure of the patch path.
        The words for RL experiments can be 'map', 'episode'.
        The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.
    rootpath : str
        The root directory path containing sample to clip from sample file path.
        Assumes file path contains rootpath.

    Returns
    -------
    str
        Sample ID generated from path.
    """
    ids = [None]*len(paths)
    for idx, path in enumerate(paths):
        ids[idx] = create_sample_id(path,
                sample_pattern=sample_pattern,
                rootpath=rootpath)
    return ids

def label_from_id(sample_id, word, sample_pattern):
    """Get label corresponding to word from sample ID.

    Parameters
    ----------
    sample_id : str
        Sample ID get label from
    word : str
        Word to the label corresponds to.
    sample_pattern : dict of (str: int)
        Dictionary describing the structure of the patch ID.
        The words for RL experiments can be 'map', 'episode'.
        The words for cancer experiments can be 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'.

    Returns
    -------
    int
        Patch size
    """
    return int(sample_id.split('/')[sample_pattern[word]])    

def index_ids(ids, sample_pattern, include=[], exclude=[]):
    """Index IDs by sample pattern words.
    For example if sample_pattern of IDs is 'annotation/subtype/slide/patch_size/magnification' and we have IDs like

    Stroma/MMRd/VOA-1000A/512/20/0_0
    Stroma/MMRd/VOA-1000A/512/10/0_0
    Stroma/MMRd/VOA-1000A/512/20/2_2
    Stroma/MMRd/VOA-1000A/256/20/0_0
    Stroma/MMRd/VOA-1000A/256/10/0_0
    Tumor/POLE/VOA-1000B/256/10/0_0

    Setting include=['patch_size'] gives

    512/0_0: [
        Stroma/MMRd/VOA-1000A/512/20/0_0
        Stroma/MMRd/VOA-1000A/512/10/0_0
    ],
    512/2_2: [
        Stroma/MMRd/VOA-1000A/512/20/2_2
    ],
    256/0_0: [
        Stroma/MMRd/VOA-1000A/256/20/0_0
        Stroma/MMRd/VOA-1000A/256/10/0_0
        Tumor/POLE/VOA-1000B/256/10/0_0
    ]

    So here we create meta IDs of form 'patch_size/patch_id' that sample IDs are grouped into.
    Setting exclude=['patch_size', 'magnification'] gives

    Stroma/MMRd/VOA-1000A/0_0: [
        Stroma/MMRd/VOA-1000A/512/20/0_0
        Stroma/MMRd/VOA-1000A/512/10/0_0
        Stroma/MMRd/VOA-1000A/256/20/0_0
        Stroma/MMRd/VOA-1000A/256/10/0_0
    ],
    Stroma/MMRd/VOA-1000A/2_2: [
        Stroma/MMRd/VOA-1000A/512/20/2_2
    ],
    Tumor/POLE/VOA-1000B: [
        Tumor/POLE/VOA-1000B/256/10/0_0
    ]

    Parameters
    ----------
    patch_ids : list of str

    sample_pattern : dict
        Dictionary describing the directory structure of the patch paths.
        The words are 'annotation', 'subtype', 'slide', 'patch_size', 'magnification'

    include : iterable of str
        The words to group by. By default includes all words.
    
    exclude : iterable of str
        The words to exclude.

    Returns
    -------
    dict of str: list
        The patch IDs grouped by words.
    """
    id_nd = np.array([[*id.split('/'), id] for id in ids], dtype=np.dtype('U'))
    words = set(sample_pattern) - set(exclude)
    if include:
        words = words & set(include)
    indices = sorted([sample_pattern[word] for word in words] + [
            id_nd.shape[1] - 2, id_nd.shape[1] - 1])
    id_nd = id_nd[:,indices]
    id_nd = np.apply_along_axis(lambda r: np.array(['/'.join(r[:-1]), r[-1]]),
            1, id_nd)
    group = { }
    for common_id, id in id_nd:
        if common_id not in group:
            group[common_id] = []
        group[common_id].append(id)
    return group

class NumpyEncoder(json.JSONEncoder):
    """The encoding object used to serialize np.ndarrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_datum(datum, directory, filename):
    """Save datum (dict with ndarray values as JSON file).

    Parameters
    ----------
    datum : dict
        The data to save
    directory : str
        The directory name to save the data.
    filename : str
        The file name to name the data.
        If file name contains '/' separated words then create subfolders for them.
        A filename will have the .json suffix added to them if necessary.
    """
    if not os.path.isdir(directory):
        raise UtilityException(f"{directory} does not exist.")
    if filename.startswith('/'):
        raise UtilityException(f"filename {filename} cannot begin with a '/'.")

    """Create subfolders if necessary"""
    filepath = os.path.join(directory, filename)
    os.makedirs(get_dirname_of(filepath), exist_ok=True)

    """Save the file"""
    filepath = filepath if filepath.endswith('.json') else f"{filepath}.json"
    with open(filepath, 'w') as f:
            json.dump(datum, f, cls=NumpyEncoder)
