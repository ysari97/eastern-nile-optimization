from model_nile import ModelNile

nile_model = ModelNile()


def nile_wrapper(decision_variables):
    objectives = nile_model.evaluate(decision_variables)
    return list(objectives)
