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

    if "documents" in prepared_kwargs and "labels" in prepared_kwargs:
        PRELOADED_DATASET = (prepared_kwargs["documents"], prepared_kwargs["labels"])

    # use floats in argstring to load results from main
    if args["maxsum_factor"]==1:
        args["maxsum_factor"]= 1.0
    if args["mmr_param"]==1:
        args["mmr_param"]= 1.0

    args = args.items()
    argstring = str(sorted(args))
    #load = False # REMOVE THIS!!
    score = main.FormFillingIterator(**prepared_kwargs, load = load, save = save, argstring = argstring, fields_length = fields_length, mode=mode)()

    wandb.log(score)

best_choice_params = {
        "ff_model" :{"value" : "best_choice"},
        "context_shortener" :{"value" : "retrieval"},
        "chunk_info_to_compare" : {"values": [
            "direct",
            #"keybert",
            ]},
        "field_info_to_compare": {"value":"choices"},
    }

fullpaper_params = { # best baseline
        "ff_model" :{"value" : "4om"},
        "context_shortener" : {"value" :"full_paper"},
    }

gpt_sota= {
        "ff_model" :{"values" : ["4om"]},
        "field_info_to_compare" : {"values":[
            "choices",
            "choice-list",
            ]},
        "include_choice_every" : {"values" :[
            1,
            3,
            5,
            8,
            ]},
        "similarity_k" : {"values": [10]},
        }
gpt_rag_params = {
        "ff_model" :{"values" : [
            "4om",
            ]},
        "field_info_to_compare" : {"values":[
            "description",
            ]},
        "similarity_k" : {"values": [10]},
        }
        # TODO set context length
        #llama_sota= {
        #        "ff_model" :{"values" : ["hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"]},
        #        "field_info_to_compare" : {"values":[
        #            "choices",
        #            "choice-list",
        #            ]},
        #        "include_choice_every" : {"values" :[
        #            1,
        #            3,
        #            5,
        #            8,
        #        "similarity_k" : {"values": [10]},
        #        }
        #mistral_sota= {
        #        "ff_model" :{"values" : ["TheBloke/Mistral-7B-v0.1-GPTQ"]},
        #        "field_info_to_compare" : {"values":[
        #            "choices",
        #            "choice-list",
        #            ]},
        #        "include_choice_every" : {"values" :[
        #            1,
        #            3,
        #            5,
        #            8,
        #        "similarity_k" : {"values": [10]},
        #        }

#open_rag_params = {
#        "ff_model" :{"values" : [
#            "TheBloke/Mistral-7B-v0.1-GPTQ",
#            "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
#            ]},
#        "field_info_to_compare" : {"values":[
#            "description",
#            ]},
#        "similarity_k" : {"values": [10]},
#        }

test_11 = {
        "ff_model" :{"value" : "4om"},
        "field_info_to_compare" : {"values":[
            "choices",
            "choice-list",
            "description",
            ]},
        "similarity_k" : {"values": [10]},
        "dataset_shuffle" : {"value" : 11},
        }






#
#keyword_llama_params = {
#        "context_shortener" : {"values" : [
#            "keyword-literal", 
#            "keybert-literal", 
#            ]},
#        "chunk_size" : {"values" : [
#            #150,
#            #200,
#            300,
#            #500, 
#            ]},
#        "chunk_overlap" : {"value" : 100},
#        "similarity_k" : {"values" : [
#            #2,
#            #3,
#            4,
#            #5,
#            ]},
#    }
#
#rag_llama_params = {
#        "embedding_model" :{"values": [
#            #"Losspost/stella_en_1.5b_v5",
#            "all-minilm:l6-v2",
#            ]},
#        "chunk_size" : {"values" : [
#            #150,
#            #200,
#            #300, # more is clearly better
#            500,  
#            #700,  
#            #850,
#            #1000,
#            ]},
#        "chunk_overlap" : {"value" : 100},
#        "similarity_k" : {"values" : [
#            #2,
#            #3,
#            #4, # more is clearly better 
#            5,
#            #6,
#            #7,
#            #8,
#            ]},
#    }


def run_sweep(parameters, dataset_length=0, sweep_count=1, method="grid", dataset = "arxpr", name = None, fields_length = 0, mode = "train"):
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
    fl = 100
    run_sweep(best_choice_params, 
              fields_length = fl,
              sweep_count = 1,
              dataset=["arxpr2"],
              mode = "test",
              name="best_choice",
              )
    run_sweep(gpt_rag, 
              fields_length = fl,
              sweep_count = 1,
              dataset=["arxpr2"],
              mode = "test",
              name="gpt_rag",
              )
    run_sweep(gpt_sota, 
              fields_length = fl,
              sweep_count = 8,
              dataset=["arxpr2"],
              mode = "test",
              name="gpt_sota",
              )
    quit()
    run_sweep(fullpaper_params, 
              fields_length = fl,
              sweep_count = 1,
              dataset=["arxpr2"],
              mode = "test",
              name="gpt_fullpaper",
              )
if __name__ == "__main__":
    run_tests()
    quit()
    run_sweep(sota_11, 
              dataset_length=300,
              fields_length = 30,
              sweep_count = 1,
              dataset=["arxpr2"],
              name="newpipeline_notshufled",
              )
    run_sweep(sota_params, 
              dataset_length=300,
              fields_length = 30,
              sweep_count = 1,
              dataset=["arxpr2"],
              name="newpipeline_notshufled",
              )
    quit()
    run_sweep(best_choice_params, 
              fields_length = 30,
              sweep_count = 3,
              dataset=["arxpr2"],
              name="newpipelinetest",
              )
    run_sweep(fullpaper_params, 
              fields_length = 30,
              sweep_count = 3,
              dataset=["arxpr2"],
              name="newpipelinetest",
              )
    quit()
    run_sweep(sota_params, 
              fields_length = 30,
              sweep_count = 3,
              dataset=["arxpr2"],
              name="newpipelinetest",
              )
    quit()
    #TODO cleanup or remove below
    run_sweep(shorter_literal_test, 
              dataset_length = 300,
              fields_length = 30,
              sweep_count = 5,
              method = "grid",
              dataset=["arxpr2s:1_25", "arxpr2s:2_25", "arxpr2s:3_25", "arxpr2s:4_25", "arxpr2s_25"],
              name="reshufled_lit5_kw",
              )
    quit()

    run_sweep(kw_test, 
              dataset_length = 300,
              fields_length = 30,
              sweep_count = 5,
              method = "grid",
              dataset=["arxpr2s:1_25", "arxpr2s:2_25", "arxpr2s:3_25", "arxpr2s:4_25", "arxpr2s_25"],
              name="reshufled_kw",
              )
    quit()
    run_sweep(shorter_literal_test, 
              dataset_length = 300,
              fields_length = 30,
              sweep_count = 8,
              method = "grid",
              dataset=["arxpr2s_25"],
              name="shorterliteral",
              )
    quit()
    run_sweep(new_rag_params_2, 
              dataset_length = 300,
              fields_length = 30,
              sweep_count = 2,
              method = "grid",
              dataset=["arxpr2s_25"],
              name="newrags",
              )
    #quit()
    #quit()
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
