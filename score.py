import numpy as np
import sklearn.metrics

def calc_score(solution,submission):
    scores = []
    for component in range(solution.shape[1]):
        y_true_subset = solution[:, component]
        y_pred_subset = submission[:, component]
        scores.append(sklearn.metrics.recall_score(
            y_true_subset, y_pred_subset, average='macro'))
    final_score = np.average(scores, weights=[2,1,1])
    return final_score
