import wandb
import dspy
from argparse import Namespace
import yaml

import main


model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
dspy_model = dspy.HFModel(model = model_id)



def add_defaults(parameters):

    with open("arguments.yaml", "r") as f:
        argument_template = yaml.safe_load(f)

    for argtype in argument_template:
        for argname in argument_template[argtype]:
            if not argname in parameters:
                parameters[argname] = {
                        "value" : argument_template[argtype][argname]["default"]
                        }
    return parameters



def sweep_run():
    wandb.init(project= "upcast_profiler")
    
    args = wandb.config

    prepared_kwargs = main.load_modules(args, preloaded_dspy_model = dspy_model)
    args = args._items
    args.pop("_wandb")
    args.pop("dataset_length")
    load = args.pop("load")
    save = args.pop("save")
    args = args.items()
    argstring = str(sorted(args))
    score = main.fill_out_forms(**prepared_kwargs, argstring = argstring, load=load, save=save)

    wandb.log(score)

dk_params = {
        "ff_model" :{"value" : "keybert"},
        "context_shortener" : {"value" : "keybert-literal"},
        "reduce_chunk_size" : {"values" : [
            500, 
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
        "n_keywords" : {"value" : 8},
    }

openai_params = {
        "ff_model" :{"value" : "4om"},
        "context_shortener" : {"values" : [
            "full_paper",
            ]},
    }

rag_params = {
        "ff_model" :{"value" : "4om"},
        "context_shortener" : {"values" : [
            "rag", 
            ]},
        "rag_chunk_size" : {"values" : [512,]},
        "rag_chunk_overlap" : {"value" : 64},
        "similarity_k" : {"values" : [3,]},
    }

keybert_params = {
        "context_shortener" : {"values" : [
            "keybert-literal", 
            ]},
        "ff_model" :{"value" : "4om"},
        "reduce_chunk_size" : {"values" : [
            500, 
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
        "similarity_k" : {"values" : [
            #4,
            14,
            #16
            ]},
        "n_keywords" : {"values" : [
            8
            ]},
    }


def run_sweep(parameters, dataset_length, sweep_count, method, dataset = "arxpr", name = None):
    parameters["dataset_length"] = {"value" : dataset_length}
    parameters["dataset"] = {"value" : dataset}
    parameters = add_defaults(parameters)
    
    
    sweep_configuration = {
        "name":f"{name}_{dataset}_{sweep_count}_{dataset_length}",
        "method": method, # random, grid (every config) or bayesian
        "metric": {"goal": "maximize", "name": "total_score"},
        "parameters": parameters,
    }
    
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="upcast_profiler")

    wandb.agent(sweep_id, function=sweep_run, count=sweep_count)
    #wandb.teardown()

if __name__ == "__main__":

    for name, params in [("direct_kw", dk_params), ("4om", openai_params), ("kw-4om", keybert_params), ("rag-4om", rag_params)]:
        run_sweep(params, 
                  dataset_length = 100,
                  sweep_count = 1,
                  method = "grid",
                  dataset = "arxpr2_100",
                  name=name,
                  )

