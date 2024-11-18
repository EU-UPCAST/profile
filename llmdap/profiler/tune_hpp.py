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
            #500, 
            #1000, 
            2000, 
            #5000
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
        "dk_norm_order" : {"values" : [1,2,10]},
        "n_keywords" : {"value" : 8},
    }

keybert_vs_rag= {
        "ff_model" :{"value" : "4om"},
        "context_shortener" : {"values" : [
            #"keybert-both",
            "rag", 
            ]},
        "rag_chunk_size" : {"values" : [512,]},
        "reduce_chunk_size" : {"values" : [
            #500, 
            #1000, 
            2000, 
            #5000
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
        "similarity_k" : {"values" : [3,]},
        "n_keywords" : {"value" : 8},
    }

keybert_params = {
        "context_shortener" : {"values" : [
            #"keybert-label",
            "keybert-description", 
            "keybert-literal", 
            #"keybert-both"
            ]},
        "ff_model" :{"value" : "4om"},
        "reduce_chunk_size" : {"values" : [
            #250
            500, 
            #1000, 
            #2000, 
            #5000
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
        "similarity_k" : {"values" : [
            #10,
            #12,
            14,
            #16
            ]},#4,6,8,10]},#,2,3,]},
        "n_keywords" : {"values" : [
            #5, 
            8
            ]},#,12]},
    }

openai_parameters = {
        "ff_model" :{"values" : [
            "4om",
            # add non-mini version
            ]},
        "context_shortener" : {"values" : ["full_paper"]},
    }

rag_parameters = {
        #"sampler_beams" : {"values" : [2,3,4]},
        "context_shortener" : {"values" : ["rag"]},
        #"retriever_type" : {"values" : ["simple","fusion"]},
        "rag_chunk_size" : {"values" : [
            #128,
            #256,
            512,
            #1024,
            #2048,
            #4096
            ]},
        "similarity_k" : {"values" : [
            #1,
            #2,
            3,
            5,
            ]},
        "mmr_param" : {"values" : [
            #0.1,
            #0.4,
            0.7,
            0.9,
            1,
            ]},
        "rag_llm" : {"values" : [
            #"biolm",
            "llama3.1",
            #"text-embedding-3-small",
            #"text-embedding-3-large",
            ]},
        "ff_model" : {"values" : [
            #"biolm",
            "llama3.1I-8b-q4",
            #3"4om",
            ]},
    }

reduce_parameters = {
        "sampler_beams" : {"values" : [2,4]},
        "context_shortener" : {"values" : ["reduce"]},
        "reduce_chunk_size" : {"values" : [5000, 10000,20000,30000]},
        "reduce_temperature" : {"values" : [0.01, 0.15, 0.4, 0.9]},
    }



def run_sweep(parameters, dataset_length, sweep_count, method, dataset = "arxpr"):
    parameters["dataset_length"] = {"value" : dataset_length}
    parameters["dataset"] = {"value" : dataset}
    parameters = add_defaults(parameters)
    
    
    sweep_configuration = {
        "method": method, # random, grid (every config) or bayesian
        "metric": {"goal": "maximize", "name": "total_score"},
        "parameters": parameters,
    }
    
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="upcast_profiler")

    wandb.agent(sweep_id, function=sweep_run, count=sweep_count)
    #wandb.teardown()

if __name__ == "__main__":

    #run_sweep(dk_params, 
    #          dataset_length = 10,
    #          sweep_count = 4,
    #          method = "grid",
    #          dataset = "study_type",
    #          )
    run_sweep(dk_params, 
              dataset_length = 100,
              sweep_count = 3,
              method = "grid",
              dataset = "study_type",
              )

