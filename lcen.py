# Implementation of the LCEN algorithm


def lasso_clip_en(
        X,
        y,
        alpha: list,
        l1_ratio: list,
        degree: list,
        lag: list,
        cutoff
):
    l1_ratio = 1

    # Work through permutations of alpha, degree, and lag
    # perform cross-validation
    # return best model

    #TODO implement cross-validation

    best_degree = None
    best_lag = None

    #CLIP Step
    # Remove all parameters < cutoff

    #EN Step
    # Restore l1_ratio to original list of values

    # For each combination in (alpha, l1_ratio)
    # Perform cross validation

    #TODO Implement cross-validation

    # Clip Step II
    # Remove all parameters < cutoff