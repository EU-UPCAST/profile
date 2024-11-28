import wandb
import dspy
from argparse import Namespace
import yaml

import main


#model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
#dspy_model = dspy.HFModel(model = model_id)
dspy_model = None

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
    mode = args.pop("mode")
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
    score = main.fill_out_forms(**prepared_kwargs, argstring = argstring, load=load, save=save, fields_length=fields_length, mode=mode)

    wandb.log(score)

dk_params = { # direct keyword
        "ff_model" :{"value" : "keybert"},
        "context_shortener" : {"value" : "keyword-literal"},
        "reduce_chunk_size" : {"value" : 500},
        "reduce_chunk_overlap" : {"value" : 100},
    }

openai_params = { # best baseline
        "ff_model" :{"value" : "4om"},
        "context_shortener" : {"value" :"full_paper"},
    }

rag_params = { # baseline rag
        "ff_model" :{"value" : "4om"},
        "rag_llm" :{"values": [
            #"llama3.1",
            #"Losspost/stella_en_1.5b_v5",
            "all-minilm:l6-v2",
            ]},
        "context_shortener" : {"values" : [
            "rag", 
            ]},
        "rag_chunk_size" : {"values" : [
            #100,
            700, 850, 1000]},
        #"rag_chunk_size" : {"values" : [500]},
        "rag_chunk_overlap" : {"value" : 100}, # ?!?!?
        "similarity_k" : {"values" : [
            #10, 
            17]},
    }


keyword_llama_params = {
        "context_shortener" : {"values" : [
            "keyword-literal", 
            "keybert-literal", 
            ]},
        "reduce_chunk_size" : {"values" : [
            #150,
            #200,
            300,
            #500, 
            ]},
        "reduce_chunk_overlap" : {"value" : 100},
        "similarity_k" : {"values" : [
            #2,
            #3,
            4,
            #5,
            ]},
    }

rag_llama_params = { # TODO tune this.
        "rag_llm" :{"values": [
            #"Losspost/stella_en_1.5b_v5",
            "all-minilm:l6-v2",
            ]},
        "rag_chunk_size" : {"values" : [
            #150,
            #200,
            #300, # more is clearly better
            500,  
            ]},
        "rag_chunk_overlap" : {"value" : 100},
        "similarity_k" : {"values" : [
            #2,
            #3,
            #4, # more is clearly better 
            5,
            ]},
    }
kw_test = {
        "context_shortener" : {"values" :[
            "keyword-literal", # best overall (on train 50)
            "keybert-literal", # best on anything but hardware (on train 50)
            ]},
        "ff_model" :{"value" : "4om"},
        "reduce_chunk_size" : {"value" : 500},
        "reduce_chunk_overlap" : {"value" : 100},
        "similarity_k" : {"value" : 10},
    }


def run_sweep(parameters, dataset_length, sweep_count, method, dataset = "arxpr", name = None, fields_length = 0, mode = "train"):
    parameters["dataset_length"] = {"value" : dataset_length}
    parameters["fields_length"] = {"value" : fields_length}
    parameters["mode"] = {"value" : mode}
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

def run_tests():
    dl, fl = 700, 50
    run_sweep(openai_params, 
              dataset_length = dl,
              fields_length = fl,
              sweep_count = 1,
              method = "grid",
              dataset=["arxpr2s_25"],
              mode = "test",
              name="openai_test",
              )
    run_sweep(dk_params, 
              dataset_length = dl,
              fields_length = fl,
              sweep_count = 1,
              method = "grid",
              dataset=["arxpr2s_25"],
              mode = "test",
              name="dk_test",
              )
    run_sweep(kw_test, 
              dataset_length = dl,
              fields_length = fl,
              sweep_count = 2,
              method = "grid",
              dataset=["arxpr2s_25"],
              mode = "test",
              name="kw_test",
              )
    run_sweep(keyword_llama_params, 
              dataset_length = dl,
              fields_length = fl,
              sweep_count = 2,
              method = "grid",
              dataset=["arxpr2s_25"],
              mode = "test",
              name="kw_llama_test",
              )
    quit()
    run_sweep(rag_params, 
              dataset_length = dl,
              fields_length = fl,
              sweep_count = 2,
              method = "grid",
              dataset=["arxpr2s_25"],
              mode = "test",
              name="rag_test",
              )

if __name__ == "__main__":
    #run_tests()
    #quit()

    #run_sweep(rag_llama_params, 
    #          dataset_length = 300,
    #          fields_length = 30,
    #          sweep_count = 16,
    #          method = "random",
    #          dataset=["arxpr2s_25"],
    #          name="rag_llama_tune",
    #          )
    #quit()
    run_sweep(rag_params, 
              dataset_length = 300,
              fields_length = 30,
              sweep_count = 3,
              method = "grid",
              dataset=["arxpr2s_25"],
              name="rag_retune",
              )
    quit()
    #run_sweep(dk_params, 
    #          dataset_length = 600,
    #          fields_length = 30,
    #          sweep_count = 6,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          name="dk_return",
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
