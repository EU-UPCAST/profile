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
import openai

import dataset_loader
import metadata_schemas 
import form_filling
import evaluation
import context_shortening

import nltk
nltk.download('averaged_perceptron_tagger_eng')


def set_openai_api_key():
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


def save_score(argstring, scores, index_log, choice_log, dataset):
    if not argstring: # something wrong
        raise ValueError
    try:
        with open(f"all_results/{dataset}_scores.json") as f:
            data = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        data= {"scores":{}, "index_logs":{}, "choice_logs":choice_log}
    data["scores"][argstring] = scores
    data["index_logs"][argstring] = index_log
    if not "choice_log" in data:
        data["choice_log"] = {}
    data["choice_log"][argstring] = choice_log
    os.makedirs("all_results", exist_ok = True)
    with open(f"all_results/{dataset}_scores.json", "w") as f:
        json.dump(data, f)





class FormFillingIterator:
    def __init__(
        self,
        context_shortener, 
        form_filler, 
        documents=None, 
        form_generator=None,
        document_generator=None,
        labels=None, 
        evaluation_fnc=None, 
        remove_fields = lambda x:[], 
        argstring="", 
        save=True, 
        load = True, 
        fields_length = 0, 
        mode = "train",
        dataset_name = ""):

        # make sure we have correct inputs
        if documents is None:
            assert labels is None
            assert not form_generator is None
            assert not document_generator is None
            assert fields_length>0
        else:
            assert not form_filler.pydantic_form is None
            assert form_generator is None
            assert document_generator is None

        self.context_shortener = context_shortener
        self.form_filler = form_filler
        self.documents = documents
        self.form_generator = form_generator
        self.document_generator = document_generator
        self.labels = labels
        self.evaluation_fnc = evaluation_fnc
        self.remove_fields = remove_fields
        self.argstring = argstring
        self.save = save
        self.load = load
        self.fields_length = fields_length
        self.mode = mode
        self.dataset_name = dataset_name

        self.field_names = self.form_filler.pydantic_form.__fields__ 

        self.all_scores = {}
        self.index_log = {}
        self.choice_log = {}
        for field in self.field_names:
            self.all_scores[field] = []
            self.index_log[field] = []
            self.choice_log[field] = []
        #self.all_times = []
        self.skips = 0

        if documents is None:
            self.iterate = self._iterate_using_generator
        else:
            self.iterate = self._iterate_using_list

    @weave.op()
    def __call__(self):
        self.iterate()

        if self.documents is None or self.labels:
            return self.evaluate()
        return

    def _iterate_using_generator(self):

        while True:
            key, paper_labels = self.document_generator.get_next_labels()

            # get equal amount of predictions for each label
            # by removing labels for fields with enough predictions already
            skipped_fields = 0
            for field in paper_labels:
                if len(self.all_scores[field]) >= self.fields_length:
                    #if len(paper_labels[field]):
                    #    print("--- skipping field with enough preidictions:", field)
                    paper_labels[field] = []
                    skipped_fields += 1
            if skipped_fields >= len(self.all_scores): # all fields have the required number of predictions
                print("---------Enough predictions made")
                break

            if len(paper_labels) == len(self.remove_fields(paper_labels)):
                #print("!!! No usable labels, skippping paper")
                continue

            # now that most labels have been disgarded (via continue), we load document (which takes a bit of time)
            paper_text = self.document_generator.get_paper_text(key)

            pydantic_form = self.form_generator(seed = int(key)) # use key as seed to ensure unique seeds
            self.form_filler.re_set_pydantic_form(pydantic_form)

            filled_form = self.fill_single_form(key,paper_text, paper_labels)

            # log index
            self.log_index(filled_form, paper_labels, pydantic_form)

    def log_index(self,filled_form, paper_labels, pydantic_form):

        pydantic_fields = pydantic_form.__fields__
        for fieldname in paper_labels:
            if len(paper_labels[fieldname]):
                assert len(paper_labels[fieldname]) == 1

                literal_values = pydantic_fields[fieldname].annotation.__args__

                label_choice = paper_labels[fieldname][0]
                label_index = literal_values.index(label_choice)

                pred_choice = getattr(filled_form, fieldname)
                pred_index = literal_values.index(pred_choice)

                print((label_index, pred_index))
                self.index_log[fieldname].append((label_index, pred_index))
                self.choice_log[fieldname].append((label_choice, pred_choice))


    def _iterate_using_list(self):
        # iterate through documents
        for docnr, key in enumerate(self.documents):

            print("loading doc", key, ", nr", docnr, "/", len(self.documents))
            #start_time = time.time()
            paper_text = self.documents[key]
            if self.labels:
                paper_labels = self.labels[key]

                if self.fields_length:
                    # get equal amount of predictions for each label
                    # by removing labels for fields with enough predictions already
                    skipped_fields = 0
                    for field in paper_labels:
                        if len(self.all_scores[field]) >= self.fields_length:
                            #if len(paper_labels[field]):
                            #    print("--- skipping field with enough preidictions:", field)
                            paper_labels[field] = []
                            skipped_fields += 1
                    if skipped_fields >= len(self.all_scores): # all fields have the required number of predictions
                        print("---------Enough predictions made")
                        break

                if len(paper_labels) == len(self.remove_fields(paper_labels)):
                    #print("!!! No usable labels, skippping paper")
                    continue
            
                self.fill_single_form(key,paper_text, paper_labels)
            else:
                self.fill_single_form(key,paper_text)


    @weave.op()
    def fill_single_form(self, key, paper_text, paper_labels=None):
        pydantic_form = self.form_filler.pydantic_form

        filled_form = None

        if self.load:
            filled_form = load_form(key, self.argstring, pydantic_form)

            if not (filled_form is None or filled_form == "skipped"):
                # check all fields with labels have been filled
                field_missing = False 
                for field in filled_form.__fields__:
                    label = paper_labels[field]
                    # each paper only have labels for a subset of the fields.
                    # we only calculate score for these
                    if len(label):
                        pred = getattr(filled_form, field)
                        if pred is None:
                            field_missing = True
                            print("misssing field: ", field)
                if field_missing:
                    print("!! unloading document due to missing field(s)!!")
                    filled_form = None # un-load


        if filled_form is None or filled_form == "skipped":
        
            print("--------- setting document")
            self.context_shortener.set_document(paper_text)
        
            # fill out the form
            try:
                print("--------- generating")
                if not paper_labels is None:
                    filled_form = self.form_filler.forward(self.context_shortener, exclude_fields=self.remove_fields(paper_labels))
                else:
                    filled_form = self.form_filler.forward(self.context_shortener)
                save_form(key, self.argstring, filled_form.dict())
            except torch.OutOfMemoryError:
                print("OUT of memory, skipping")
                self.skips += 1
                save_form(key, self.argstring, "skipped")
                return
            except openai.BadRequestError as m:
                print("!! BAD REQUEST ERROR!!")
                print(m)
                print("skipping paper and filling zeros in score")
                # evaluate
                if self.labels:
                    for field in list(set(paper_labels.keys())-set(self.remove_fields(paper_labels))):
                        print("zeroing field:", field)
                        self.all_scores[field].append(0)
                    #all_times.append(time.time()-start_time)
                    return

        elif filled_form == "skipped":
            skips += 1
            return


        # evaluate
        if not paper_labels is None:
            scores = self.evaluation_fnc(paper_labels, filled_form, verbose=False)

            print("score:", scores)
            print("\n")
            for field in scores:
                self.all_scores[field].append(scores[field])
            #all_times.append(time.time()-start_time)

            #print number of scores in each field
            for fn in self.all_scores:
                print(fn, len(self.all_scores[fn]), end= "; ")
            print("")

        return filled_form

    def evaluate(self):

        if self.mode == "test":
            save_score(self.argstring, self.all_scores, self.index_log, self.choice_log, dataset = self.dataset_name)
        #print("________printing final scores:")
        #pprint.pprint(all_scores)
        means_by_field = {}
        for field in self.all_scores:
            print(field, np.mean(self.all_scores[field]))
            means_by_field[field] = np.mean(self.all_scores[field])

        # calculate mean score
        final_score = []
        final_accuracy = []
        final_similarity = []
        for field in self.all_scores:
            scores = self.all_scores[field]
            final_score.extend(scores)

            field_properties = self.form_filler.pydantic_form.schema()["properties"][field]
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
        print("all scores:", final_score)
        print("length:", len(final_score))
        print("mean", np.mean(final_score))
        
        info_to_log = means_by_field
        info_to_log["final_scores"] = final_score
        info_to_log["total_score"] = np.mean(final_score)
        info_to_log["total_accuracy"] = np.mean(final_accuracy)
        info_to_log["total_similarity"] = np.mean(final_similarity)
        #info_to_log["seconds"] = np.mean(all_times)
        info_to_log["papers_skipped"] = self.skips

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
def load_modules(args, preloaded_dspy_model = None, preloaded_dataset = None):
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
    use_best_choice_generator = False
    if args.ff_model == "4om": # openai model
        model_id = "gpt-4o-mini"
        model_is_openai = True
        set_openai_api_key()
    elif args.ff_model == "4o": # openai model
        model_id = "gpt-4o"
        model_is_openai = True
        set_openai_api_key()
    elif args.ff_model == "best_choice":
        use_best_choice_generator = True
    elif args.ff_model == "None": # do not load any model (used for retrieval evaluation)
        model_id = ""
        model_is_openai = True
    else: # huggingface model, with outlines
        # load HF llm through dspy
        if preloaded_dspy_model is None:
            model_kwargs = {}
            if args.ff_model == "llama3.1I-8b-q4":
                model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
            elif args.ff_model == "biolm":
                model_id = "aaditya/Llama3-OpenBioLLM-8B"
            elif args.ff_model == "ministral_gguf":
                model_id = "bartowski/Ministral-8B-Instruct-2410-GGUF"
                model_kwargs = {"gguf_file" : "Ministral-8B-Instruct-2410-Q4_K_M.gguf"}
            else:
                model_id = args.ff_model
            try:
                    dspy_model = dspy.HFModel(model = model_id, hf_device_map = "cuda:2", model_kwargs = model_kwargs)
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




    if args.dataset == "arxpr2" and args.dataset_shuffle == "r":
        # do dynamic reloading+shuffling
        length = args.dataset_literal_length
        form_generator = metadata_schemas.get_shuffled_arxpr2(length = length)
        document_generator = dataset_loader.Arxpr_generator(version = "2_25", mode=args.mode)
        dataset_kwargs = dict(
                form_generator = form_generator,
                document_generator = document_generator,
                )
        pydantic_form = form_generator()
    elif args.dataset == "study_type" and args.dataset_shuffle == "r":
        # do dynamic reloading+shuffling
        length = args.dataset_literal_length
        form_generator = metadata_schemas.get_shuffled_arxpr2(length = length, only_shuffle_type = True)
        document_generator = dataset_loader.Studytype_generator(version = "2_25", mode=args.mode)
        dataset_kwargs = dict(
                form_generator = form_generator,
                document_generator = document_generator,
                )
        pydantic_form = form_generator(0)
    else:
        # load up front
        loader_kwargs = {"max_amount": args.dataset_length}
        if args.dataset == "arxpr":
            loader = dataset_loader.load_arxpr_data
            pydantic_form = metadata_schemas.arxpr_schema 
        elif args.dataset == "arxpr2":
            loader_kwargs["version"] = "2_25" # loaded dataset always 25, only pydantic form depends on literal_length and shuffling
            if args.dataset_shuffle == "s": #preshuffled
                raise NotImplementedError # just shiffle here instead
                #pydantic_form = metadata_schemas.arxpr2s_schemas[str(args.dataset_literal_length)] # TODO shuffle
            elif args.dataset_shuffle == "n": # not shuffled
                pydantic_form = metadata_schemas.arxpr2_schemas[str(args.dataset_literal_length)]
            elif type(args.dataset_shuffle) is int: #preshuffled
                length = args.dataset_literal_length
                form_generator = metadata_schemas.get_shuffled_arxpr2(length = length)
                pydantic_form = form_generator(seed=args.dataset_shuffle)
            else:
                raise ValueError

            loader_kwargs["mode"] = args.mode #train or test
            loader = dataset_loader.load_arxpr_data
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
        if preloaded_dataset is None:

            all_documents, all_labels = loader(**loader_kwargs)
        else:
            all_documents, all_labels = preloaded_dataset

        dataset_kwargs = dict(
                documents = all_documents,
                labels = all_labels,
                )


    # set context_shortener
    if args.context_shortener == "rag":
        context_shortener = context_shortening.RAGShortener(
                embed_model = args.embedding_model,
                pydantic_form = pydantic_form,
                retriever_type = args.retriever_type,
                chunk_size = args.chunk_size,
                chunk_overlap = args.chunk_overlap,
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
                chunk_size = args.chunk_size,
                chunk_overlap = args.chunk_overlap,
                max_tokens = args.reduce_max_tokens,
                )
    elif args.context_shortener == "full_paper":
        context_shortener = context_shortening.FullPaperShortener()
    elif args.context_shortener == "retrieval":
        if not args.dataset in ["study_type", "arxpr2"]:# keybert-based retrieval implementation requires all fields are literals, or have ontology, for now # TODO update so description retrieval is viable
            raise ValueError

        context_shortener = context_shortening.Retrieval(
                chunk_info_to_compare = args.chunk_info_to_compare,
                field_info_to_compare = args.field_info_to_compare,
                include_choice_every = args.include_choice_every,
                embedding_model_id = args.embedding_model,
                pydantic_form = pydantic_form, # TODO change this with the shuffling
                n_keywords = args.n_keywords,
                top_k = args.similarity_k,
                chunk_size = args.chunk_size,
                chunk_overlap = args.chunk_overlap,
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
        elif args.context_shortener in ["rag", "retrieval"]:
            form_filler = form_filling.OpenAISequentialFormFiller(
                    model_id=model_id,
                    pydantic_form = pydantic_form,
                    listify_form = args.listed_output,
                    max_tokens = args.openai_ff_max_tokens,
                    verbose=False)
        else:
            raise NotImplementedError
    elif use_best_choice_generator:
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
    if args.remove_fields == "None":
        remove_fields = lambda x:[]
    elif args.remove_fields == "empty":
        remove_fields = remove_empty_fields
    elif args.remove_fields == "non-single":
        remove_fields = remove_non_single_fields
    else:
        print(args.remove_fields)
        raise ValueError



    prepared_kwargs = dict(
            context_shortener = context_shortener,
            form_filler = form_filler,
            evaluation_fnc=evaluation.score_general_prediction,
            remove_fields = remove_fields,
            **dataset_kwargs
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
    args.pop("dataset_length")
    mode = args.pop("mode")
    fields_length = args.pop("fields_length")
    argstring = str(sorted(args.items()))
    FormFillingIterator(**prepared_kwargs, load = load, save = save, argstring = argstring, fields_length = fields_length, mode=mode, dataset_name = args.dataset)()
