import numpy as np


def cost_of_label(data, decision, classification, data_type='point'):
    """
    This function gives back the cost of obtaining a label from the incoming data. It will check if the classification
    of the points one wants to investigate is correct and assesses a profit or loss on each result.

    a data point has the format [x,y,label] where label is always the last entry in the list

    Parameters
    ----------
    data : can be either a data point or a array of data points.
    decision : bool or array, investigate a point True or False
    classification : label of data point given by model.
    data_type : is the incoming data a point or an array of data/batch

    Returns
    -------
    costs for obtaining a certain label of a data point or labels of an array of data points

    """
    if data_type == 'point':
        if decision == 1:
            # for pointlike input
            cost = cost_investigation(data, classification)
            return cost
        else:
            return 0.0

    elif data_type == 'array':
        # array like data input
        df = pd.DataFrame(list(data), columns={'x', 'y', 'label'})
        df['pred_label'] = classification
        df_invest = df[decision == 1].copy()
        df['cost'] = df_invest.apply(lambda x: cost_investigation(x[['x', 'y', 'label']], x['pred_label']), axis=1)
        df.fillna(0, inplace=True)

        return np.array(df['cost'])


def cost_investigation(data_point, predicted_label, salary=-1.0, fraud_label=1):
    """
    Function that looks at the data is there fraud or not compared to what the model predicts.
    We want to obtain the labels, did we make a correct decission or not and the cost consequences are made
    Parameters
    ----------
    data_point : series or array like object that has as last entry the true label
    predicted_label : classification label given by the model
    salary : how much does a standard investigation cost, default = -1
    fraud_label : default we have determined fraud at 1

    Returns
    -------
    cost of investigating for a label

    """
    correct_label = data_point[-1]
    profit = 0
    loss = 0
    if (correct_label == predicted_label) & (correct_label == fraud_label):
        # we have found fraud correctly
        # we are on the right way to get our investigation back
        profit = 2
        loss = 0
    elif (correct_label == predicted_label) & (correct_label != fraud_label):
        # we did not find fraud in a case where we did not expect fraud
        # so we will qualify this as breaking even
        profit = 1
        loss = 0
    elif (correct_label != predicted_label) & (correct_label == fraud_label):
        # we found fraud where the algorithm actually predicted no fraud
        # hmm strange, it is good that we investigated it!!
        profit = 2
        loss = 0
    elif (correct_label != predicted_label) & (correct_label != fraud_label):
        # we have predicted fraud but it was not fraud at all
        # hmm we are on the wrong path
        profit = 0
        loss = -1
    cost = salary + profit + loss
    return cost

