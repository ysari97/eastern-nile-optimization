from model_nile import ModelNile

nile_model = ModelNile()

total_parameter_count = nile_model.overarching_policy.get_total_parameter_count()
release_parameter_count = nile_model.overarching_policy.functions["release"].get_free_parameter_number()
n_inputs_release = nile_model.overarching_policy.functions["release"].n_inputs
n_outputs_release = nile_model.overarching_policy.functions["release"].n_outputs
RBF_count = nile_model.overarching_policy.functions["release"].RBF_count
p_per_RBF = 2 * n_inputs_release + n_outputs_release

# Since we first introduce the release policy to the model, first parameters
# belong to the RBFs. Only then, the parameters of hedging functions.
# Let's first put the levers for the release policy.

lever_list = list()
for i in range(release_parameter_count):
    modulus = (i - n_outputs_release) % p_per_RBF
    if (
        (i >= n_outputs_release)
        and (modulus < (p_per_RBF - n_outputs_release))
        and (modulus % 2 == 0)
    ):  # centers:
        lever_list.append([-1, 1])
    else:  # linear parameters for each release, radii and weights of RBFs:
        lever_list.append([0, 1])

for j in range(release_parameter_count, total_parameter_count):
    lever_list.append([0, 1])

def nile_wrapper(decision_variables):
    objectives = nile_model.evaluate(decision_variables)
    # print(objectives, flush=True)
    return list(objectives)
    
