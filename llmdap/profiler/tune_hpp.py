import wandb
import dspy
from argparse import Namespace
import yaml

import main


model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
dspy_model = dspy.HFModel(model = model_id)

PRELOADED_DATASET = None


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
    global PRELOADED_DATASET
    wandb.init(project= "upcast_profiler")
    
    args = wandb.config


    prepared_kwargs = main.load_modules(args, preloaded_dspy_model = dspy_model, preloaded_dataset = PRELOADED_DATASET)
    args = args._items
    args.pop("_wandb")
    args.pop("dataset_length")
    load = args.pop("load")
    save = args.pop("save")
    fields_length = args.pop("fields_length")

    PRELOADED_DATASET = (prepared_kwargs["documents"], prepared_kwargs["labels"])

    # use floats in argstring to load results from main
    if args["maxsum_factor"]==1:
        args["maxsum_factor"]= 1.0
    if args["mmr_param"]==1:
        args["mmr_param"]= 1.0

    args = args.items()
    argstring = str(sorted(args))
    #load = False # REMOVE THIS!!
    score = main.fill_out_forms(**prepared_kwargs, argstring = argstring, load=load, save=save, fields_length=fields_length)

    wandb.log(score)

dk_params = { # direct keyword
        "ff_model" :{"value" : "keybert"},
        "context_shortener" : {"value" : "keyword-literal"},
        "reduce_chunk_size" : {"values" : [
            500, 
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
    }

openai_params = { # best baseline
        "ff_model" :{"value" : "4om"},
        "context_shortener" : {"values" : [
            "full_paper",
            ]},
    }

rag_params = { # baseline rag
        "ff_model" :{"value" : "4om"},
        "rag_llm:" :{"values": [
            #"Losspost/stella_en_1.5b_v5",
            "all-minilm:l6-v2",
            #"llama3.1",
            ]},
        "context_shortener" : {"values" : [
            "rag", 
            ]},
        "rag_chunk_size" : {"values" : [500,]},
        "rag_chunk_overlap" : {"value" : 100},
        #"similarity_k" : {"values" : [14,15,16,17,18,19,20]},
        "similarity_k" : {"values" : [10, 15]},
    }

keybert_params = { # best baseline
        "context_shortener" : {"values" : [
            "keybert-literal", 
            ]},
        "ff_model" :{"value" : "4om"},
        "reduce_chunk_size" : {"values" : [
            500, 
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
        "similarity_k" : {"values" : [
            12,
            13,
            14,
            15,
            16,
            #17,
            #18,
            ]},
        "n_keywords" : {"values" : [
            8
            ]},
    }

keyword_llama_params = { # still tuning
        "context_shortener" : {"values" : [
            "keyword-literal", 
            ]},
        "reduce_chunk_size" : {"values" : [
            500, 
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
        "similarity_k" : {"values" : [
            2,
            3,
            4,
            6,
            9,
            12,
            16
            ]},
    }

keyword_llama_params_2 = {
        "context_shortener" : {"values" : [
            #"keyword-literal", 
            "keybert-literal", 
            ]},
        "reduce_chunk_size" : {"values" : [
            #150,
            200,
            #300,
            #400,
            #500, 
            #800,
            #1200,
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
        "similarity_k" : {"values" : [
            #2,
            3,
            #4,
            #5,
            #6,
            #9,
            #12,
            #16
            ]},
    }
kw_vs_kb_params= { # yes keyword has still been run with all these params
        "context_shortener" : {"values" : [
            #"keybert-literal-4", 
            #"keybert-literal-3", 
            #"keybert-literal", 
            "keyword-literal", 
            ]},
        "ff_model" :{"value" : "4om"},
        "reduce_chunk_size" : {"values" : [
            500, 
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
        "similarity_k" : {"values" : [
            9,
            11,
            8,
            12,
            10,
            14,
            15,
            #16
            ]},
        "n_keywords" : {"values" : [
            8
            ]},
    }

def run_sweep(parameters, dataset_length, sweep_count, method, dataset = "arxpr", name = None, fields_length = 0):
    parameters["dataset_length"] = {"value" : dataset_length}
    parameters["fields_length"] = {"value" : fields_length}
    if type(dataset) is str:
        parameters["dataset"] = {"value" : dataset}
        name = f"{name}_{dataset}_{sweep_count}_{dataset_length}"
    if type(dataset) is list:
        parameters["dataset"] = {"values" : dataset}
        name = f"{name}__{sweep_count}_{dataset_length}.{fields_length}"
    parameters = add_defaults(parameters)
    
    
    sweep_configuration = {
        "name": name,
        "method": method, # random, grid (every config) or bayesian
        "metric": {"goal": "maximize", "name": "total_score"},
        "parameters": parameters,
    }
    
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="upcast_profiler")

    wandb.agent(sweep_id, function=sweep_run, count=sweep_count)
    #wandb.teardown()

if __name__ == "__main__":
    #run_sweep(dk_params, 
    #          dataset_length = 1000,
    #          fields_length = 30,
    #          sweep_count = 1,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          name="new_dk",
    #          )
    #run_sweep(rag_params, 
    #          dataset_length = 1000,
    #          fields_length = 30,
    #          sweep_count = 2,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          name="rag_tune",
    #          )
    #run_sweep(keybert_params, 
    #          dataset_length = 1000,
    #          fields_length = 30,
    #          sweep_count = 5,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          name="kb_tune",
    #          )
    run_sweep(kw_vs_kb_params, 
              dataset_length = 725,
              fields_length = 50,
              sweep_count = 6,
              method = "grid",
              dataset=["arxpr2s_25"],
              name="keybert_or_keyword",
              )
    #run_sweep(keyword_llama_params_2, 
    #          dataset_length = 500,
    #          sweep_count = 5,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],#, "arxpr2s_50", "arxpr2s_100", "arxpr2s_200", "arxpr2s_400"],
    #          name="tuned_llama_500papers",
    #          )
    #run_sweep(keyword_llama_params_2, 
    #          dataset_length = 1000,
    #          fields_length = 30,
    #          sweep_count = 1,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],#, "arxpr2s_50", "arxpr2s_100", "arxpr2s_200", "arxpr2s_400"],
    #          name="tuned_llama_1kpapers",
    #          )
    #run_sweep(keybert_params, 
    #          dataset_length = 100,
    #          sweep_count = 5,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],#, "arxpr2s_50", "arxpr2s_100", "arxpr2s_200", "arxpr2s_400"],
    #          name="ragtuneK",#"tuned_llama_25->400",
    #          )


    #for p in [openai_params, keybert_params]: #
    #for p in [rag_params]:
    #    run_sweep(p, 
    #              dataset_length = 100,
    #              sweep_count = 1,
    #              method = "grid",
    #              dataset="arxpr2s_25",
    #              name="2s25-4omresults",
    #              )
