import torch
import pyro

def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=0)

def add_bias(model, bias_map):
    def _add_bias_model(*args, **kwargs):
        trace = pyro.poutine.trace(model).get_trace(*args, **kwargs)
        for name, site in trace.nodes.items():
            if name in bias_map:
                site["log_prob"] = site["log_prob"] + bias_map[name] * site["value"]
        return trace.log_prob_sum()
    return _add_bias_model

def counterfactual_query(model, factual_trace, intervention, query):
    # Abduction: infer latent variables
    conditioned_model = pyro.poutine.condition(model, data=factual_trace)
    inferred_trace = pyro.poutine.trace(conditioned_model).get_trace()

    # Action: apply intervention
    intervened_model = pyro.poutine.do(conditioned_model, data=intervention)

    # Prediction: query the intervened model
    return pyro.poutine.trace(intervened_model).get_trace()