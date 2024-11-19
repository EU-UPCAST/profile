import numpy as np
import torch
import weave
import pprint
import outlines
import transformers
import dspy
import copy
import pydantic
from typing import Optional
import argparse
import time
import json
import os

import dataset_loader
import metadata_schemas 
import form_filling
import evaluation
import context_shortening

import nltk
nltk.download('averaged_perceptron_tagger_eng')


def set_openai_api_key():
    import openai
    from openai_key import API_KEY
    openai.api_key = API_KEY


def make_optional_model(model: pydantic.BaseModel) -> pydantic.BaseModel:
    """
    Make a new pydantic model where all the fields are optional.
    Node that string contraints, descriptions and examples are not kept
    (this is not needed for evaluation)
    """
    fields = {name: (Optional[field.annotation], None) for name, field in model.__fields__.items()}
    optional_model = pydantic.create_model(f'{model.__name__}Optional', **fields)
    return optional_model


def load_form(key, argstring, pydantic_form):
    if not argstring: # something wrong
        raise ValueError
    try:
        with open("all_results/"+key+".json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("------ file not found")
        return None # file does not exist
    except json.decoder.JSONDecodeError as e:
        print("------ file corrupted(?)")
        print(e)
        return None # corrupted - loading fail
    try:
        data = data[argstring]
    except KeyError:
        return None # file does not contain a run with these arguments - loading fails
    if type(data) == str and data == "skipped":
        return "skipped"
    optional_form = make_optional_model(pydantic_form)
    print("------ load successfull")
    return optional_form(**data)

def save_form(key, argstring, form_dict):
    if not argstring: # something wrong
        raise ValueError
    try:
        with open("all_results/"+key+".json") as f:
            data = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        data= {}
    data[argstring] = form_dict
    os.makedirs("all_results", exist_ok = True)
    with open("all_results/"+key+".json", "w") as f:
        json.dump(data, f)




@weave.op()
def fill_out_forms(documents, context_shortener, form_filler, labels=None, evaluation_fnc=None, remove_fields = lambda x:[], argstring="", save=True, load = True):

    all_scores = {}
    for field in form_filler.pydantic_form.__fields__:
        all_scores[field] = []
    all_times = []
    skips = 0
    # iterate through documents
    for docnr, key in enumerate(documents):
        print("loading doc", key, ", nr", docnr, "/", len(documents))
        start_time = time.time()
        paper_text = documents[key]
        if labels:
            paper_labels = labels[key]
            if len(paper_labels) == len(remove_fields(paper_labels)):
                print("!!! No usable labels, skippping paper")
                continue
            if "arxpr2_" in argstring:
                # skip the common ones just to avoid having a very skewed score. TODO: solve this problem in a better way
                if set(paper_labels.keys())-set(remove_fields(paper_labels)) <= {"assay_by_molecule_14", "study_type_18"}:
                    print("!!! only the common labels, skippping paper")
                    continue

        filled_form = None

        if load:
            filled_form = load_form(key, argstring, form_filler.pydantic_form)


        if filled_form is None:
    
            print("--------- setting document")
            context_shortener.set_document(paper_text)
    
            # fill out the form
            try:
                print("--------- generating")
                if labels:
                    filled_form = form_filler.forward(context_shortener, exclude_fields=remove_fields(paper_labels))
                else:
                    filled_form = form_filler.forward(context_shortener)
                #print("---------labels:")
                #pprint.pprint(paper_labels)
                #print("---------form:")
                #printform(filled_form)
                #print("---------")
                save_form(key, argstring, filled_form.dict())
            except torch.OutOfMemoryError:
                print("OUT of memory, skipping")
                skips += 1
                save_form(key, argstring, "skipped")
                continue
        elif filled_form == "skipped":
            skips += 1
            continue


        # evaluate
        if labels:
            scores = evaluation_fnc(paper_labels, filled_form, verbose=False)

            print("score:", scores)
            print("\n")
            for field in scores:
                all_scores[field].append(scores[field])
            all_times.append(time.time()-start_time)


    if labels:
        #print("________printing final scores:")
        #pprint.pprint(all_scores)
        means_by_field = {}
        for field in all_scores:
            print(field, np.mean(all_scores[field]))
            means_by_field[field] = np.mean(all_scores[field])

        # calculate mean score
        final_score = []
        final_accuracy = []
        final_similarity = []
        for field in all_scores:
            scores = all_scores[field]
            final_score.extend(scores)

            field_properties = form_filler.pydantic_form.schema()["properties"][field]
            if (
                    field_properties["type"] == "integer" or
                    (field_properties["type"] == "string" and "enum" in field_properties)
                    ):
                print(field, " -- accuacy")
                final_accuracy.extend(scores)
            else:
                final_similarity.extend(scores)
                print(field, " -- similarity ")
            #print(field_properties["type"], "enum" in field_properties, field_properties)
        print(final_score)
        print(np.mean(final_score))
        
        info_to_log = means_by_field
        info_to_log["final_scores"] = final_score
        info_to_log["total_score"] = np.mean(final_score)
        info_to_log["total_accuracy"] = np.mean(final_accuracy)
        info_to_log["total_similarity"] = np.mean(final_similarity)
        info_to_log["seconds"] = np.mean(all_times)
        info_to_log["papers_skipped"] = skips

        return info_to_log



def remove_non_single_fields(labels):
    return [field for field in labels.keys() if len(labels[field]) != 1]

def remove_empty_fields(labels):
    return [field for field in labels.keys() if len(labels[field]) == 0]

def printform(filled_form):
    fields = filled_form.schema()["properties"]
    for key, value in filled_form.dict().items():
        fields[key].update({"value": value})
        del fields[key]["type"]
        del fields[key]["title"]
    pprint.pprint(fields, width=170)

@weave.op() # log args
def load_modules(args, preloaded_dspy_model = None):
    """
    prepare arguments, then call fill_out_forms 
    preloaded_dspy_model can be inputted to avoid loading it in memory several times
    """

    # log arguments
    if args.log:
        weave.init(project_name = "upcast_profiler",
                   )

    # load llm
    model_is_openai = False
    direct_keybert = False
    if args.ff_model == "4om": # openai model
        model_id = "gpt-4o-mini"
        model_is_openai = True
        set_openai_api_key()
    elif args.ff_model == "4o": # openai model
        model_id = "gpt-4o"
        model_is_openai = True
        set_openai_api_key()
    elif args.ff_model == "keybert":
        direct_keybert = True
    elif args.ff_model == "None": # do not load any model (used for retrieval evaluation)
        model_id = ""
        model_is_openai = True
    else: # huggingface model, with outlines
        # load HF llm through dspy
        if preloaded_dspy_model is None:
            if args.ff_model == "llama3.1I-8b-q4":
                model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
            elif args.ff_model == "biolm":
                model_id = "aaditya/Llama3-OpenBioLLM-8B"
            else:
                model_id = args.ff_model
            try:
                dspy_model = dspy.HFModel(model = model_id, hf_device_map = "cuda:3")
            except RuntimeError:
                dspy_model = dspy.HFModel(model = model_id, hf_device_map = "cuda:0")
        else:
            dspy_model = preloaded_dspy_model
        hf_model = dspy_model.model
        hf_tokenizer = dspy_model.tokenizer

        # set some dspy model options
        #dspy_model.kwargs["max_tokens"]=args.max_tokens
        dspy_model.drop_prompt_from_output = True

        # define outlines llm and sampler
        outlines_llm = outlines.models.Transformers(model=hf_model, tokenizer=hf_tokenizer)
        if args.sampler == "greedy":
            outlines_sampler = outlines.samplers.GreedySampler()
        elif args.sampler == "beam":
            outlines_sampler = outlines.samplers.BeamSearchSampler(beams = args.sampler_beams)
        elif args.sampler == "multi":
            outlines_sampler = outlines.samplers.MultinomialSampler(
                    top_k=args.sampler_top_k,
                    top_p=args.sampler_top_k,
                    temperature=args.sampler_temp,
                    )
        else:
            raise ValueError


    # load data
    if args.dataset == "arxpr":
        loader = dataset_loader.load_arxpr_data
        pydantic_form = metadata_schemas.arxpr_schema 
    elif args.dataset[:6] == "arxpr2":
        dataset_literal_length = args.dataset.split("2_")[1]
        def loader(max_amount=10):
            return dataset_loader.load_arxpr_data(max_amount, version = "2_25") # loaded dataset always 25, only pydantic form depends on literal_length
        pydantic_form = metadata_schemas.arxpr2_schemas[dataset_literal_length]
    elif args.dataset == "study_type":
        loader = dataset_loader.load_study_type_data
        pydantic_form = metadata_schemas.study_type_schema 
    elif args.dataset == "ega":
        loader = dataset_loader.load_ega_data
        pydantic_form = metadata_schemas.ega_schema
    elif args.dataset == "nhrf":
        loader = dataset_loader.load_nhrf_examples
        pydantic_form = metadata_schemas.nhrf_qa_schema
    elif args.dataset == "nhrf2":
        loader = dataset_loader.load_nhrf_examples2
        pydantic_form = metadata_schemas.nhrf_schema
    elif args.dataset == "nhrf3":
        loader = dataset_loader.load_nhrf_examples3
        pydantic_form = metadata_schemas.nhrf_qa_schema_2
    elif args.dataset == "simple_test":
        loader = dataset_loader.get_simple_test
        pydantic_form = metadata_schemas.arxpr_schema 
    else:
        raise ValueError
    all_documents, all_labels = loader(args.dataset_length)


    # set context_shortener
    if args.context_shortener == "rag":
        ## set RAG options
        if args.rag_llm == "llama3.1":
            chat_model = "llama3.1:8b"
            embed_model = "llama3.1:8b"
        elif args.rag_llm == "biolm":
            chat_model = "koesn/llama3-openbiollm-8b"
            embed_model = "koesn/llama3-openbiollm-8b"
        elif args.rag_llm in ["text-embedding-3-small", "text-embedding-3-large"]:
            chat_model = args.rag_llm
            embed_model = args.rag_llm
        else:
            raise NotImplementedError
        context_shortener = context_shortening.RAGShortener(
                chat_model,
                embed_model,
                pydantic_form,
                retriever_type = args.retriever_type,
                chunk_size = args.rag_chunk_size,
                chunk_overlap = args.rag_chunk_overlap,
                similarity_k = args.similarity_k,
                mmr_param = args.mmr_param,
                )
    elif args.context_shortener == "rerank":
        if model_is_openai:
            raise NotImplementedError
        context_shortener = context_shortening.Rerank(hf_model, hf_tokenizer)
    elif args.context_shortener == "reduce":
        if model_is_openai:
            raise NotImplementedError
        context_shortener = context_shortening.Reduce(
                hf_model,
                hf_tokenizer,
                temperature = args.reduce_temperature,
                chunk_size = args.reduce_chunk_size,
                chunk_overlap = args.reduce_chunk_overlap,
                max_tokens = args.reduce_max_tokens,
                )
    elif args.context_shortener == "full_paper":
        context_shortener = context_shortening.FullPaperShortener()
    elif args.context_shortener[:8] == "keybert-":
        if not args.dataset[:7] in ["study_t", "arxpr2_"]: # keybert requires all fields are literals, or have ontology
            raise ValueError
        context_shortener = context_shortening.Keybert(
                args.context_shortener.split("-")[1],
                pydantic_form = pydantic_form,
                n_keywords = args.n_keywords,
                top_k = args.similarity_k,
                chunk_sizes = (args.reduce_chunk_size, args.reduce_chunk_overlap),
                mmr_param = args.mmr_param,
                maxsum_factor = args.maxsum_factor,
                keyphrase_range = (args.keyphrase_min, args.keyphrase_min + args.keyphrase_range_diff),
                )
    else:
        print(args.context_shortener)
        raise ValueError


    if model_is_openai:
        if args.context_shortener=="full_paper":
            form_filler = form_filling.OpenAIFormFiller(
                    model_id=model_id,
                    pydantic_form = pydantic_form,
                    listify_form = args.listed_output,
                    max_tokens = args.openai_ff_max_tokens,
                    verbose=False)#True)
        elif args.context_shortener=="rag" or args.context_shortener[:8]=="keybert-":
            form_filler = form_filling.OpenAISequentialFormFiller(
                    model_id=model_id,
                    pydantic_form = pydantic_form,
                    listify_form = args.listed_output,
                    max_tokens = args.openai_ff_max_tokens,
                    verbose=False)
        else:
            raise NotImplementedError
    elif direct_keybert:
        form_filler = form_filling.DirectKeywordSimilarityFiller(
                pydantic_form=pydantic_form,
                listify_form=args.listed_output,
                verbose=False)

    else:
        # load llm form filler
        form_filler = form_filling.SequentialFormFiller(
                outlines_llm,
                outlines_sampler,
                pydantic_form=pydantic_form,
                listify_form=args.listed_output,
                answer_in_quotes=args.answer_in_quotes,
                max_tokens = args.outlines_ff_max_tokens)

    # remove fields?
    if args.remove_fields == "None" or (args.remove_fields == "default" and len(all_labels)==0):
        remove_fields = lambda x:[]
    elif args.remove_fields == "empty":
        remove_fields = remove_empty_fields
    elif args.remove_fields == "non-single" or (args.remove_fields=="default" and len(all_labels)):
        remove_fields = remove_non_single_fields


    prepared_kwargs = dict(
            documents = all_documents, 
            context_shortener = context_shortener,
            form_filler = form_filler,
            labels=all_labels, 
            evaluation_fnc=evaluation.score_general_prediction,
            remove_fields = remove_fields,
            )
    return prepared_kwargs



def parse_terminal_arguments():

    import yaml
    with open("arguments.yaml", "r") as f:
        argument_template = yaml.safe_load(f)

    parser = argparse.ArgumentParser()

    for argtype in argument_template:
        for argname in argument_template[argtype]:
            arg_info = argument_template[argtype][argname]
            if type(arg_info["default"]) == bool:
                arg_info["action"] = argparse.BooleanOptionalAction
            else:
                arg_info["type"] = type(arg_info["default"])
            if not "help" in arg_info:
                arg_info["help"] = ""
            arg_info["help"] = "( in " + argtype + ") : " + arg_info["help"]
            parser.add_argument("--"+argname, **arg_info)
    args = parser.parse_args()
    return args

if __name__ == "__main__":


    args = parse_terminal_arguments()
    prepared_kwargs = load_modules(args)
    args = args.__dict__
    load = args.pop("load")
    save = args.pop("save")
    save = args.pop("dataset_length")
    argstring = str(sorted(args.items()))
    fill_out_forms(**prepared_kwargs, load = load, save = save, argstring = argstring)


