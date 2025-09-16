def extract_predictions(predictions_, conf_thresh, label_names=None):
    if "label_names" in predictions_:
        predictions_class = list(predictions_["label_names"])
    else:
        labels = [int(i) for i in list(predictions_.get("labels", []))]
        if label_names:
            predictions_class = [
                str(label_names[i]) if 0 <= int(i) < len(label_names) else str(int(i))
                for i in labels
            ]
        else:
            predictions_class = [str(int(i)) for i in labels]
    if len(predictions_class) < 1:
        return [], [], []

    # Get the predicted bounding boxes
    predictions_boxes = [
        [(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])
    ]
    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    # Get a list of index with score greater than threshold
    threshold = float(conf_thresh)
    # predictions_t = [
    #     predictions_score.index(x) for x in predictions_score if x > threshold
    # ]
    predictions_t = [
        idx for idx, s in enumerate(predictions_score) if float(s) > threshold
    ]
    if len(predictions_t) <= 0:
        return [], [], []
    # predictions in score order
    predictions_boxes = [predictions_boxes[i] for i in predictions_t]
    predictions_class = [predictions_class[i] for i in predictions_t]
    predictions_scores = [predictions_score[i] for i in predictions_t]
    return predictions_class, predictions_boxes, predictions_scores
