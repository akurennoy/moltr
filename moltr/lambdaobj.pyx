cimport cython
cimport openmp
from cython.parallel import prange

from libc.math cimport fabs
from libc.stdlib cimport malloc, free
from libc.string cimport memset

from argsort cimport argsort


cdef double get_sigmoid(
    double score, 
    double* sigmoid_table, 
    int n_sigmoid_bins,
    double min_sigmoid_arg,
    double max_sigmoid_arg,
    double sigmoid_idx_factor
) nogil:
    if score <= min_sigmoid_arg:
        return sigmoid_table[0]
    elif score >= max_sigmoid_arg:
        return sigmoid_table[n_sigmoid_bins - 1]
    else:
        return sigmoid_table[<int>((score - min_sigmoid_arg) * sigmoid_idx_factor)]


cdef void get_gradients_for_one_query(
    double* gains, 
    double* preds, 
    long start, 
    long end,
    double* grad, 
    double* hess, 
    double* discounts,
    double inverse_max_dcg,
    double* sigmoid_table,
    long n_sigmoid_bins,
    double min_sigmoid_arg,
    double max_sigmoid_arg,
    double sigmoid_idx_factor
) nogil:
    cdef int cnt
    cnt = <int>(end - start)
    cdef int* sorted_idx = <int*> malloc(cnt * sizeof(int))

    argsort((&preds[0]) + start, sorted_idx, cnt)

    cdef double best_score, worst_score
    best_score = preds[start + sorted_idx[0]]
    worst_score = preds[start + sorted_idx[cnt - 1]]
    cdef int should_adjust 
    if best_score != worst_score:
        should_adjust = 1
    else:
        should_adjust = 0
    
    cdef double p_labmda, p_hess
    cdef double gain_high, gain_low
    cdef double score_high, score_low, delta_score
    cdef double abs_delta_ndcg, paired_discount, gain_diff
    cdef int high, low

    cdef int i
    cdef int j
    for i in range(cnt):
        high = sorted_idx[i]
        gain_high = gains[start + high]
        score_high = preds[start + high]
        for j in range(cnt):
            low = sorted_idx[j]
            gain_low = gains[start + low]
            score_low = preds[start + low]
            if gain_high > gain_low:
                delta_score = score_high - score_low
                p_lambda = get_sigmoid(delta_score,
                                       sigmoid_table,
                                       n_sigmoid_bins,
                                       min_sigmoid_arg,
                                       max_sigmoid_arg,
                                       sigmoid_idx_factor)
                p_hess = p_lambda * (2.0 - p_lambda)

                gain_diff = gain_high - gain_low
                paired_discount = fabs(discounts[1 + j] - discounts[1 + i])
                abs_delta_ndcg = gain_diff * paired_discount * inverse_max_dcg
                if should_adjust == 1:
                    abs_delta_ndcg /= (0.01 + fabs(delta_score))
                    
                p_lambda *= -abs_delta_ndcg
                p_hess *= 2 * abs_delta_ndcg

                grad[start + high] += p_lambda
                hess[start + high] += p_hess
                grad[start + low] -= p_lambda
                hess[start + low] += p_hess
    free(sorted_idx)


cdef void _get_gradients(
    double* gains, 
    double* preds,
    long n_preds,
    long* groups,
    long* query_boundaries,
    long n_queries,
    double* discounts,
    double* inverse_max_dcgs,
    double* sigmoid_table,
    long n_sigmoid_bins,
    double min_sigmoid_arg,
    double max_sigmoid_arg,
    double sigmoid_idx_factor,
    double* grad,
    double* hess
) nogil:
    memset(grad, 0, n_preds * sizeof(double))
    memset(hess, 0, n_preds * sizeof(double))

    cdef double inverse_max_dcg
    cdef long start, end, i

    for i in prange(n_queries, nogil=True):
        start = query_boundaries[i]
        end = query_boundaries[i + 1]
        inverse_max_dcg = inverse_max_dcgs[i]
        get_gradients_for_one_query(gains, 
                                    preds, 
                                    start, 
                                    end,
                                    grad, 
                                    hess, 
                                    discounts,
                                    inverse_max_dcg,
                                    sigmoid_table,
                                    n_sigmoid_bins,
                                    min_sigmoid_arg,
                                    max_sigmoid_arg,
                                    sigmoid_idx_factor)


def get_gradients(double[::1] gains, 
                  double[::1] preds,
                  long n_preds,
                  long[::1] groups,
                  long[::1] query_boundaries,
                  long n_queries,
                  double[::1] discounts,
                  double[::1] inverse_max_dcgs,
                  double[::1] sigmoid_table,
                  long n_sigmoid_bins,
                  double min_sigmoid_arg,
                  double max_sigmoid_arg,
                  double sigmoid_idx_factor,
                  double[::1] grad,
                  double[::1] hess):
    _get_gradients((&gains[0]),
                   (&preds[0]),
                   n_preds,
                   (&groups[0]),
                   (&query_boundaries[0]),
                   n_queries,
                   (&discounts[0]),
                   (&inverse_max_dcgs[0]),
                   (&sigmoid_table[0]),
                   n_sigmoid_bins,
                   min_sigmoid_arg,
                   max_sigmoid_arg,
                   sigmoid_idx_factor,
                   (&grad[0]),
                   (&hess[0]))