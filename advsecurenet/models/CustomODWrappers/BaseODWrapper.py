import numpy as np

class BaseODWrapper:
    def __init__(
        self,
        model,
        conf_thresh,
    ):
        self.model=model
        self.conf_thresh=conf_thresh

    def filter_boxes(self, predictions):
        dictionary = {}
        boxes_list = []
        scores_list = []
        labels_list = []
        for i in range(len(predictions["boxes"])):
            score = predictions["scores"][i]
            if score >= self.conf_thresh:
                boxes_list.append(predictions["boxes"][i])
                scores_list.append(predictions["scores"][[i]])
                labels_list.append(predictions["labels"][[i]])
        if len(boxes_list)>0 and len(scores_list)>0 and len(labels_list)>0:
            dictionary["boxes"] = np.vstack(boxes_list)
            dictionary["scores"] = np.hstack(scores_list)
            dictionary["labels"] = np.hstack(labels_list)
        y = dictionary
        return y