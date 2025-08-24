# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from numbers import Integral
from warnings import warn

from typing import Sequence, Any
import polars as pl
import polars.selectors as cs

import numpy as np
from numpy import dot
from numpy.linalg import svd
from scipy.linalg import eigh
import skbio 

from skbio.table._tabular import _create_table, _create_table_1d, _ingest_table
from skbio.stats.ordination._ordination_results import OrdinationResults
from skbio.stats.ordination._utils import scale
from skbio.stats.ordination._cutils import f_matrix_inplace_cy

from skbio.stats.ordination._principal_coordinate_analysis import _fsvd

def ingest_lazy(lazy_df, schema = None, sample_ids = None, feature_ids = None):

    feature_table = lazy_df.select(cs.numeric().drop_nans().drop_nulls())

    if not schema:
        schema = feature_table.collect_schema()
        
    column_names = [names for names, dtype in schema.items() if dtype.is_numeric()]
    print(column_names)
    n_samples = feature_table.select(pl.len()).collect().item()
    n_features = len(column_names)
    
    if feature_ids and len(feature_ids) == n_features:
        column_names = feature_ids
    
    if not sample_ids or (sample_ids and len(sample_ids) != n_samples):
        sample_ids = [f'{i + 1}' for i in range(1, n_samples + 1)]
    
    return feature_table, schema, sample_ids, column_names

def pca_lazy(
    table: pl.LazyFrame,
    method="eigh",
    dimensions = 0, # type:ignore
    inplace=False,
    normalize = False,
    seed=None,
    warn_neg_eigval=0.01,
    output_format=None,
) -> OrdinationResults:
    r"""Perform Principal Coordinate Analysis (PCA).
    PCA is an ordination method operating on sample x observation tables,
    calculated using Euclidean distances.

    Parameters
    ----------
    table : Table-like object
        The input sample x feature table.
    method : str, optional
        Matrix decomposition method to use. Default is "svd" which computes
        exact eigenvectors and eigenvalues for all dimensions. The alternates
        are 'eigh' and 'fsvd' (fast-singular value decomposition), a heuristic
        method that computes a specified number of dimensions.  Both alternates
        computes an intermediate matrix dependent on the shape of the matrix to
        speed up computational time.  
    dimensions : int or float, optional
        Dimensions to reduce the distance matrix to. This number determines how many
        eigenvectors and eigenvalues will be returned. If an integer is provided, the
        exact number of dimensions will be retained. If a float between 0 and 1, it
        represents the fractional cumulative variance to be retained. Default is 0,
        which will compute the rank bound of the table (minimum of two sides) 
    inplace : bool, optional
        If True, the input table will be centered in-place to reduce memory
        consumption, at the cost of losing the original observations. Default is False.
    seed : int or np.random.Generator, optional
        A user-provided random seed or random generator instance for method "fsvd".
        See :func:`details <skbio.util.get_rng>`.

        .. versionadded:: 0.6.3

    warn_neg_eigval : bool or float, optional
        Raise a warning if any negative eigenvalue is obtained and its magnitude
        exceeds the specified fraction threshold compared to the largest positive
        eigenvalue, which suggests potential inaccuracy in the PCA results. 
        .. versionadded:: 0.6.3

    Returns
    ----------
    OrdinationResults : 

    Notes
    -----
    This function relies on a mix of rectangular or symmetric solvers. The intermediate
    matrices computed in the symmetric solver paths are kept at the second-moment scale,
    and the eigenvalues are only scaled by n_samples - 1 at the very end. 

    Alternate methods are less numerically stable than SVD because an intermediate matrix
    is computed, which doubles the condition number.  Likely more noticeable in FSVD without
    scaling up the oversampling and iteration number and/or turning on the power method.  


    """
    f_matrix, f_schema, sample_ids, feature_ids = ingest_lazy(table)
    n_samples, n_features = f_matrix_shape = (len(sample_ids), len(feature_ids),)

    if n_samples == min(f_matrix_shape):
        raise NotImplementedError()

    if dimensions == 0:
        if method == "fsvd" and  min(f_matrix_shape) > 10:
            warn(
                "FSVD: since no value for number_of_dimensions is specified, "
                "PCA for all dimensions will be computed, which may "
                "result in long computation time if the original "
                "feature table is large and/or if number of features"
                "is similar or larger than the number of samples",
                RuntimeWarning,
            )
        dimensions = min(f_matrix_shape)
    elif dimensions < 0:
        raise ValueError(
            "Invalid operation: cannot reduce table "
            "to negative dimensions using PCA. Did you intend "
            'to specify the default value "0", which sets '
            "the number_of_dimensions equal to the "
            "number of features in the given table?"
        )
    elif dimensions > max(f_matrix_shape):
        raise ValueError(
            "Invalid operation: cannot extend past size of matrix."
        )
    elif dimensions > min(f_matrix_shape):
        warn(
            "The number of non-negative singular values / eigenvectors"
            "are bounded by the maximum possible rank of the feature table.",
            RuntimeWarning,
        )
        if (method == 'fsvd' or method == 'eigh'):
            dimensions = min(f_matrix_shape)
            warn(
                "FSVD or Eigh: The minimum of the feature_table shape will be"
                "computed instead.",
                RuntimeWarning,
            )      
    elif not isinstance(dimensions, Integral) and dimensions > 1:
        raise ValueError(
            "Invalid operation: A floating-point number greater than 1 cannot be "
            "supplied as the number of dimensions."
        )

    if warn_neg_eigval and not 0 <= warn_neg_eigval <= 1:
        raise ValueError(
            "warn_neg_eigval must be Boolean or a floating-point number between 0 "
            "and 1."
        )
    
    matrix_data = np.zeros((n_features,n_features))
    indices = np.triu_indices(n_features)

    stats = f_matrix.select([pl.col(c).mean().alias(f"{c}_mean") for c in feature_ids] + [pl.col(c).std().alias(f"{c}_std") for c in feature_ids]).collect().to_numpy().reshape(2, -1).T
    means = stats[:, 0]
    std = stats[:, 1]


    std[std == 0] = 1

    exprs = [
            (pl.col(ci) * pl.col(cj)).mean().alias(f"{ci}_{cj}")
            for i, ci in enumerate(feature_ids)
            for j, cj in enumerate(feature_ids[i:], i)
        ]
    
    flattened = f_matrix.select(exprs).collect().to_numpy()
    matrix_data[indices] = flattened
    matrix_data.T[indices] = flattened

    np.allclose(matrix_data, matrix_data.T, atol=1e-8)

    if np.isnan(matrix_data).any() or np.isinf(matrix_data).any():
        raise ValueError('nan or null val')

    np.subtract(matrix_data, np.outer(means, means), out = matrix_data)
    if np.isnan(matrix_data).any() or np.isinf(matrix_data).any():
        raise ValueError('subtract nan or null val')
    np.divide(matrix_data, np.outer(std, std), out = matrix_data)

    if np.isnan(matrix_data).any() or np.isinf(matrix_data).any():
        raise ValueError('divide nan or null val')


    if method == "eigh":
        eigvals, eigvecs = eigh(matrix_data, overwrite_a = True)
        long_method_name = (
            f"Principal Component Analysis Using Full Eigendecomposition"
        )

        # Eigh returns values in sorted ascending order so the
        # arrays are flipped to match the order in svd and fsvd
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        
    elif method == "fsvd":
        ndim = dimensions
        if 0 < dimensions < 1:
            warn(
                "FSVD: since value for number_of_dimensions is "
                "specified as float. PCA for all dimensions will be"
                "computed, which may result in long computation time"
                "if the original distance matrix is large. Consider"
                "specifying an integer value to optimize performance.",
                RuntimeWarning,
            )
            ndim = min(f_matrix_shape)
        eigvals, eigvecs = _fsvd(
            matrix_data, ndim, seed=seed)
        long_method_name = (
            "Approximate Principal Component Analysis using FSVD"
        )
    else:
        raise ValueError("PCA eigendecomposition method {} not supported.".format(method))
        
    coordinates, loadings = normalize_signs(eigvecs, in_sample_space = False)

    # Clip eigenvalues close to 0 to 0
    atol = np.finfo(float).eps * max(n_samples, n_features) * max(eigvals[0], 1.0)
    close_to_zero = np.isclose(eigvals, 0.0, atol = atol)
    eigvals[close_to_zero] = 0.0

    if method == "fsvd":
        # Since the dimension parameter, hereafter referred to as 'd',
        # restricts the number of eigenvalues and eigenvectors that FSVD
        # computes, we need to use an alternative method to compute the sum
        # of all eigenvalues, used to compute the array of proportions
        # explained. Otherwise, the proportions calculated will only be
        # relative to d number of dimensions computed; whereas we want
        # it to be relative to the number of features in the input table.

        # An alternative method of calculating th sum of eigenvalues is by
        # computing the trace of the centered feature table.
        # See proof outlined here: https://goo.gl/VAYiXx
        sum_eigvals = np.trace(matrix_data)
    else:
        # Calculate proportions the usual way
        sum_eigvals = np.sum(eigvals)
    
    # Since it is possible for eigh to return negative eigenvalues
    # because of rounding errors (PCA assumes euclidean distances
    # so it should be positive), we zero the corresponding values
    # associated with any negative eigenvalues.
    non_negative = (eigvals >= 0.0).sum()
    eigvals[non_negative:] = np.zeros(eigvals[non_negative:].shape)
    
    proportion_explained = eigvals / sum_eigvals
    if 0 < dimensions < 1:
        cumulative_variance = np.cumsum(proportion_explained)
        num_dimensions = np.searchsorted(cumulative_variance, dimensions, side="left") + 1
        # gives the number of dimensions needed to reach specified variance
        # updates number of dimensions to reach the requirement of variance.
        dimensions = num_dimensions

    eigvals = eigvals[:dimensions]
    proportion_explained = proportion_explained[:dimensions]
    loadings = loadings[:, :dimensions] # type: ignore


    num_positive = (eigvals > 0.0).sum()
    loadings[:, num_positive:] = np.zeros(loadings[:, num_positive:].shape)

    coordinates = np.empty((n_samples, dimensions))

    eigvals /= (n_samples - 1)

    return _encapsulate_pca_result(
        long_method_name,
        eigvals,
        coordinates,
        loadings,
        proportion_explained,
        sample_ids,
        feature_ids,
        output_format,
    )

def normalize_signs(x, y = None, in_sample_space=True, in_feature_space = True):

    if x.flags.c_contiguous:
        # columns of u, rows of v, or equivalently rows of u.T and v
        max_abs_u_cols = np.argmax(np.abs(x.T), axis=1)
        shift = np.arange(x.T.shape[0])
        indices = max_abs_u_cols + shift * x.T.shape[1]
        signs = np.sign(np.take(np.reshape(x.T, (-1,)), indices, axis=0))
    else:
        # rows of v, columns of u
        max_abs_v_rows = np.argmax(np.abs(x), axis=0)
        shift = np.arange(x.shape[1])
        indices = max_abs_v_rows + shift * x.shape[0]
        signs = np.sign(np.take(np.reshape(x, (-1,)), indices, axis=0))

    x *= signs[np.newaxis, :]

    if in_sample_space and in_feature_space:
        y *= signs[np.newaxis, :]
    
    if not in_sample_space:
        return y, x
    return x,y

def _encapsulate_pca_result(
    long_method_name:str,
    eigvals:np.ndarray,
    coordinates:np.ndarray,
    loadings:np.ndarray[Any],
    proportion_explained:np.ndarray,
    sample_ids:Sequence[str],
    feature_ids:Sequence[str],
    output_format,
) -> OrdinationResults:
    r"""Perform Principal Coordinate Analysis (PCA).
    PCA is an ordination method operating on sample x observation tables,
    calculated using Euclidean distances.

    Parameters
    ----------
    table : Table-like object
        The input sample x feature table.
    method : str, optional
        Matrix decomposition method to use. Default is "svd" which computes
        exact eigenvectors and eigenvalues for all dimensions. The alternates
        are 'eigh' and 'fsvd' (fast-singular value decomposition), a heuristic
        method that computes a specified number of dimensions.  Both alternates
        computes an intermediate matrix dependent on the shape of the matrix to
        speed up computational time.  
    dimensions : int or float, optional
        Dimensions to reduce the distance matrix to. This number determines how many
        eigenvectors and eigenvalues will be returned. If an integer is provided, the
        exact number of dimensions will be retained. If a float between 0 and 1, it
        represents the fractional cumulative variance to be retained. Default is 0,
        which will compute the rank bound of the table (minimum of two sides) 
    inplace : bool, optional
        If True, the input table will be centered in-place to reduce memory
        consumption, at the cost of losing the original observations. Default is False.
    seed : int or np.random.Generator, optional
        A user-provided random seed or random generator instance for method "fsvd".
        See :func:`details <skbio.util.get_rng>`.

        .. versionadded:: 0.6.3

    warn_neg_eigval : bool or float, optional
        Raise a warning if any negative eigenvalue is obtained and its magnitude
        exceeds the specified fraction threshold compared to the largest positive
        eigenvalue, which suggests potential inaccuracy in the PCA results. 
        .. versionadded:: 0.6.3
    Notes
    -----
    This function relies on a mix of rectangular or symmetric solvers. The intermediate
    matrices computed in the symmetric solver paths are kept at the second-moment scale,
    and the eigenvalues are only scaled by n_samples - 1 at the very end. 

    Alternate methods are less numerically stable than SVD because an intermediate matrix
    is computed, which doubles the condition number.  Likely more noticeable in FSVD without
    scaling up the oversampling and iteration number and/or turning on the power method.  


    """
    dimensions = eigvals.shape[0]
    axis_labels = ["PC%d" % i for i in range(1, dimensions + 1)]
    return OrdinationResults(
        short_method_name="PCA",
        long_method_name=long_method_name,
        eigvals = _create_table_1d(eigvals, index=axis_labels, backend=output_format),
        samples = _create_table(
            coordinates,
            index=sample_ids,
            columns=axis_labels,
            backend=output_format,
        ),
        features =_create_table(
            loadings,
            index=feature_ids,
            columns=axis_labels,
            backend=output_format,
        ),
        proportion_explained = _create_table_1d(
            proportion_explained, index=axis_labels, backend=output_format
        ),
    )
