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

sota_params = {
        "ff_model" :{"value" : "4om"},
        "field_info_to_compare " : {"values":[
            "choice-list",
            "description",
            #"choices",
            ]},
        "similarity_k" : {"values": [10]},
        }






# TODO cleanup
#rag_params = { # baseline rag
#        "ff_model" :{"value" : "4om"},
#        "embedding_model" :{"values": [
#            #"llama3.1",
#            #"Losspost/stella_en_1.5b_v5",
#            "all-minilm:l6-v2",
#            #"koesn/llama3-openbiollm-8b",
#            ]},
#        "context_shortener" : {"values" : [
#            "rag", 
#            ]},
#        "chunk_size" : {"values" : [
#            850
#            ]},
#        "similarity_k" : {"values" : [
#            #10, 
#            17]},
#    }
#rag_params_2 = { # baseline rag with comparable context
#        "ff_model" :{"value" : "4om"},
#        "embedding_model" :{"values": [
#            #"llama3.1",
#            #"Losspost/stella_en_1.5b_v5",
#            "all-minilm:l6-v2",
#            ]},
#        "context_shortener" : {"values" : [
#            "rag", 
#            ]},
#        "similarity_k" : {"value" : 10},
#    }
#shorter_literal_test = {
#        "context_shortener" : {"values" : [
#            #"keyword-list:12", # last iteration of 2 values (in 25) # probably to random w.r.t. shuffeling
#            #"keyword-list:8", # last iteration of 3 values (in 25) # probably to random w.r.t. shuffeling
#            #"keyword-list:5", # 5 values
#            #"keyword-list:3", # 8 values
#            #"keyword-list:2", # 12 values
#            #"keyword-list",
#            #"keyword-literal:5",
#            "keyword-literal:3",
#            #"keyword-literal:2",
#            #"keyword-literal",
#            #"keyword-fielddescription", 
#            ]},
#        "ff_model" :{"value" : "4om"},
#        "chunk_size" : {"value" : 500},
#        "chunk_overlap" : {"value" : 100},
#        "similarity_k" : {"value" : 10},
#    }
#kw_test = {
#        "context_shortener" : {"values" :[
#            "keyword-literal", # best overall (on train 50)
#            #"keybert-literal", # best on anything but hardware (on train 50)
#            ]},
#        "ff_model" :{"value" : "4om"},
#        "chunk_size" : {"value" : 500},
#        "chunk_overlap" : {"value" : 100},
#        "similarity_k" : {"value" : 10},
#    }
#
#keyword_open_params = {
#        "ff_model" :{"values" : [
#            ##"ministral_gguf",
#            #"mistralai/Ministral-8B-Instruct-2410",
#            #"PyrTools/Ministral-8B-Instruct-2410-GPTQ-128G",
#            #"shuyuej/Ministral-8B-Instruct-2410-GPTQ",
#            "TheBloke/Mistral-7B-v0.1-GPTQ",
#            ]},
#        "context_shortener" : {"values" : [
#            "keyword-literal", 
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
#rag_open_params = {
#        "ff_model" :{"values" : [
#            ##"ministral_gguf",
#            #"mistralai/Ministral-8B-Instruct-2410",
#            #"PyrTools/Ministral-8B-Instruct-2410-GPTQ-128G",
#            #"shuyuej/Ministral-8B-Instruct-2410-GPTQ",
#            "TheBloke/Mistral-7B-v0.1-GPTQ",
#            ]},
#        "embedding_model" :{"values": [
#            #"Losspost/stella_en_1.5b_v5",
#            "all-minilm:l6-v2",
#            ]},
#        "chunk_size" : {"values" : [
#            500,  
#            ]},
#        "chunk_overlap" : {"value" : 100},
#        "similarity_k" : {"values" : [
#            5,
#            ]},
#    }
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
    dl, fl = 1500, 100
    #run_sweep(new_rag_params_2, 
    #          dataset_length = dl,
    #          fields_length = fl,
    #          sweep_count = 2,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          mode = "test",
    #          name="newrag_rag_test",
    #          )
    #quit()
    #run_sweep(rag_open_params, 
    #          dataset_length = dl,
    #          fields_length = fl,
    #          sweep_count = 2,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          mode = "test",
    #          name="mstral_rag_test",
    #          )
    #run_sweep(keyword_open_params, 
    #          dataset_length = dl,
    #          fields_length = fl,
    #          sweep_count = 1,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          mode = "test",
    #          name="mstral_test",
    #          )
    #run_sweep(dk_params, 
    #          dataset_length = dl,
    #          fields_length = fl,
    #          sweep_count = 2,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          mode = "test",
    #          name="dk_test",
    #          )
    #run_sweep(openai_params, 
    #          dataset_length = dl,
    #          fields_length = fl,
    #          sweep_count = 1,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          mode = "test",
    #          name="openai_test",
    #          )
    #run_sweep(keyword_llama_params, 
    #          dataset_length = dl,
    #          fields_length = fl,
    #          sweep_count = 2,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          mode = "test",
    #          name="kw_llama_test",
    #          )
    #run_sweep(rag_llama_params, 
    #          dataset_length = dl,
    #          fields_length = fl,
    #          sweep_count = 1,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          mode = "test",
    #          name="rag_llama_test",
    #          )
    #run_sweep(rag_params, 
    #          dataset_length = dl,
    #          fields_length = fl,
    #          sweep_count = 1,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          mode = "test",
    #          name="rag_test",
    #          )
    #run_sweep(rag_params_2, 
    #          dataset_length = dl,
    #          fields_length = fl,
    #          sweep_count = 1,
    #          method= "grid",
    #          dataset=["arxpr2s_25"],
    #          mode = "test",
    #          name="rag_test",
    #          )

    #run_sweep(kw_test, 
    #          dataset_length = dl,
    #          fields_length = fl,
    #          sweep_count = 2,
    #          method = "grid",
    #          dataset=["arxpr2s_25"],
    #          mode = "test",
    #          name="kw_test",
    #          )
if __name__ == "__main__":
    #run_tests()
    #quit()
    run_sweep(sota_params, 
              dataset_length = 6,
              fields_length = 2,
              sweep_count = 2,
              method = "grid",
              dataset=["arxpr2"],
              name="newpipelinetest",
              )
    quit()
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
