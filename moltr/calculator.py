import numpy as np



def get_query_boundaries(groups):
    assert(len(groups) > 0)
    query_boundaries = [0] + list(np.cumsum(groups))
    
    return query_boundaries


def test__get_query_boundaries():
    assert(get_query_boundaries([2, 3]) == [0, 2, 5])
    assert(get_query_boundaries([1]) == [0, 1])
    assert(get_query_boundaries([1, 1]) == [0, 1, 2])


N_SIGMOID_BINS = 1024 * 1024
MIN_SIGMOID_ARG = -25
MAX_SIGMOID_ARG = 25


class Calculator:
    def __init__(self, gains, groups, k):       
        self.query_boundaries = get_query_boundaries(groups)
        self.gains = gains
        self.k = k
        self.discounts = Calculator._fill_discount_table(np.max(groups), k)
        
        print("Computing inverse_max_dcg-s..")
        self.inverse_max_dcgs = Calculator._fill_inverse_max_dcg_table(
            self.gains, 
            self.query_boundaries,
            self.discounts,
            k
        )
        
        print("Computing sigmoids..")
        self.sigmoids, self.sigmoid_idx_factor = Calculator._fill_sigmoid_table(
            N_SIGMOID_BINS, 
            MIN_SIGMOID_ARG, 
            MAX_SIGMOID_ARG
        )


    def get_sigmoid(self, score):
        if score <= MIN_SIGMOID_ARG:
            return self.sigmoids[0]
        elif score >= MAX_SIGMOID_ARG:
            return self.sigmoids[-1]
        else:
            return self.sigmoids[int((score - MIN_SIGMOID_ARG) * self.sigmoid_idx_factor)]
    

    def compute_ndcg(self, scores):
        dcgs = np.zeros(len(self.query_boundaries) - 1)
        
        for i in range(len(self.query_boundaries) - 1):
            order = np.argsort(scores[self.query_boundaries[i]:self.query_boundaries[i + 1]])[::-1]
            g = np.array(self.gains[self.query_boundaries[i]:self.query_boundaries[i + 1]])[order][:self.k]
            dcgs[i] = np.sum(g * self.discounts[1:(len(g) + 1)])
        return np.mean(dcgs * self.inverse_max_dcgs)


    @staticmethod
    def _fill_discount_table(max_group_length, k):
        discounts = np.zeros(1 + max_group_length)
        m = min(max_group_length, k)
        discounts[1:(1 + m)] = 1 / np.log2(1 + np.arange(1, m + 1))
        return discounts


    @staticmethod
    def _fill_inverse_max_dcg_table(gains, query_boundaries, discounts, k):
        inverse_max_dcgs = np.zeros(len(query_boundaries) - 1)
        
        for i in range(len(query_boundaries) - 1):
            g = np.sort(gains[query_boundaries[i]:query_boundaries[i + 1]])[::-1][:k]
            assert(len(discounts) > len(g))
            max_dcg = np.sum(g * discounts[1:(len(g) + 1)])
            assert(max_dcg > 0)
            inverse_max_dcgs[i] = 1 / max_dcg
            
        return inverse_max_dcgs


    @staticmethod
    def _fill_sigmoid_table(n_sigmoid_bins, min_sigmoid_arg, max_sigmoid_arg):
        sigmoid_idx_factor = n_sigmoid_bins / (max_sigmoid_arg - min_sigmoid_arg)
        
        sigmoids = 2.0 / (1 + np.exp(2.0 * 
                                     (np.arange(n_sigmoid_bins) 
                                         / sigmoid_idx_factor + min_sigmoid_arg)))
        
        return sigmoids, sigmoid_idx_factor


def test__calculator():
    gains = [0, 2, 1, 1, 0]
    groups = [3, 2]

    calculator = Calculator(gains, groups, 3)

    assert(np.allclose(calculator.discounts[1], 1.0))
    assert(np.allclose(calculator.discounts[2], 1.0 / np.log2(3)))
    
    assert(np.allclose(calculator.inverse_max_dcgs[1], 1.0))
    assert(np.allclose(calculator.inverse_max_dcgs[0], 1 / (2 + 1 / np.log2(3))))
    assert(
        np.allclose(
            calculator.compute_ndcg([-0.5, 1.0, 0.5, 0.5, 1.0]),
            0.5 * (1 + 1 / np.log2(3))
        )
    )
    assert(
        np.allclose(
            calculator.compute_ndcg([-0.5, 1.0, 0.5, 1.0, 0.5]),
            1.0
        )
    )

    assert(
        np.allclose(
            calculator.get_sigmoid(MIN_SIGMOID_ARG - 1),
            2.0 / (1 + np.exp(2 * MIN_SIGMOID_ARG)),
            atol=1e-6
        )
    )
    assert(
        np.allclose(
            calculator.get_sigmoid(MIN_SIGMOID_ARG),
            2.0 / (1 + np.exp(2 * MIN_SIGMOID_ARG)),
            atol=1e-6
        )
    )
    for sigmoid_arg in MIN_SIGMOID_ARG\
                        + np.random.random(3) * (MAX_SIGMOID_ARG - MIN_SIGMOID_ARG):
        assert(
            np.allclose(
                calculator.get_sigmoid(sigmoid_arg), 
                2.0 / (1 + np.exp(2 * sigmoid_arg)),
                atol=1e-6
            )
        )
    assert(
        np.allclose(
            calculator.get_sigmoid(MAX_SIGMOID_ARG),
            2.0 / (1 + np.exp(2 * MAX_SIGMOID_ARG)),
            atol=1e-6
        )
    )
    assert(
        np.allclose(
            calculator.get_sigmoid(MAX_SIGMOID_ARG + 1),
            2.0 / (1 + np.exp(2 * MAX_SIGMOID_ARG)),
            atol=1e-6
        )
    )

    calculator = Calculator(gains, groups, 1)

    assert(np.allclose(calculator.discounts[1], 1.0))
    assert(len(calculator.discounts) == 4)
    assert(calculator.discounts[-1] == 0)

    assert(np.allclose(calculator.inverse_max_dcgs[0], 0.5))
    assert(
        np.allclose(
            calculator.compute_ndcg([-0.5, 1.0, 0.5, 0.5, 1.0]),
            0.5
        )
    )



if __name__ == "__main__":
    test__get_query_boundaries()
    test__calculator()