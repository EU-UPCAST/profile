import argparse
import yaml
from types import SimpleNamespace





def add_defaults(parameters):
    """ given parameters for a run, add default values for all fields that are not included (from arguments.yaml) """
    with open("arguments.yaml", "r") as f:
        argument_template = yaml.safe_load(f)

    for argtype in argument_template:
        for argname in argument_template[argtype]:
            if not argname in parameters:
                parameters[argname] = argument_template[argtype][argname]["default"]
    return parameters

def call_inference(
        schema,
        parsed_paper_text = None,
        raw_xml_paper_text = None,
        paper_path = None,
        paper_url = None,
        graph_traversers=None,
        return_dict_with_context=True,
        traversal_problem_type = None,
        
        **kwargs):
    """
    Fill out the schema for one or several papers.

    Inputs:

    schema: pydantic form that the pipeline should filll out

    include ONE of the following four arguments:
        parsed_paper_text: paper text ready for reading
        raw_xml_paper_text: paper text in xml format (typically starting with "<?xml version="1.0" encoding="UTF-8"?>...)"
        paper_path: local path to paper file in XML format
        paper_url: url to paper file in XML format
    This argument should be either a string, a list of strings, or a dict of strings

    kwargs: Any argument from the arguments.yaml file (e.g. llm, retrievl method and parameters like chunk length)


    output:
    dictionary with the filled form and used contexts for each paper.

    """
    from load_modules import load_modules
    from run_modules import FormFillingIterator

    # prepare arguments
    parameters = add_defaults(kwargs)
    args=SimpleNamespace(**parameters)
    print(args)

    # load stuff
    prepared_kwargs = load_modules(args, inference_schema = schema, graph_traversers = graph_traversers, traversal_problem_type=traversal_problem_type)
    ff_iterator = FormFillingIterator(args, **prepared_kwargs)

    # make the argument into dictionary
    paper_argument = parsed_paper_text or raw_xml_paper_text or paper_path or paper_url
    if type(paper_argument) is str:
        paper_argument = {0: paper_argument}
    elif type(paper_argument) is list:
        paper_argument = {i:p for i,p in enumerate(paper_argument)}
    elif type(paper_argument) is dict:
        pass
    else:
        raise ValueError
    
    outputs = {}
    for key in paper_argument:
        
        # parse/load/fetch argument (to a string paper ready for the llm)
        if parsed_paper_text:
            paper_text = paper_argument[key]
        elif raw_xml_paper_text:
            import dataset_loader
            paper_text = dataset_loader.parse_raw_xml_string(paper_argument[key])
        elif paper_path:
            import dataset_loader
            paper_text = dataset_loader.load_paper_text_from_file_path(paper_argument[key])
        elif paper_url:
            import dataset_loader
            paper_text = dataset_loader.load_paper_text_from_url(paper_argument[key])
        else:
            raise ValueError

        # fill form
        outputs[key] =ff_iterator.fill_single_form(key=key, paper_text=paper_text, pydantic_form=schema, return_dict_with_context=return_dict_with_context)
        #if len(outputs)>2:
        #    break

    return outputs


def match_hf_acm_graphs():
    from hf_tag_graph import hftags_list
    from metadata_schemas.acm_ccs import Traverser, CCS_HIERARCHY, PathSchema

    output = call_inference(
            schema = PathSchema,
            parsed_paper_text = hftags_list,
            graph_traversers = Traverser(CCS_HIERARCHY),
            return_dict_with_context = False,
            traversal_problem_type = "graph2graph",
            # kwargs
            context_shortener = "full_paper",
            #ff_model = "4om",
            ff_model = "41n",
            #ff_model = "5n",
            )

def call_arxhf_to_acm_run(n=1, v2=False):
    if v2:
        from metadata_schemas.acm_ccs_v2 import Traverser, CCS_HIERARCHY, PathSchema
    else:
        from metadata_schemas.acm_ccs import Traverser, CCS_HIERARCHY, PathSchema

    from dataset_loader import Arxiv_HF_datasets
    ahd = Arxiv_HF_datasets()
    ahd.prepare()

    hf, arx = ahd.get_dict_format(n)

    hf_description_type = "tags and model card of a Huggingface model"
    arx_description_type = "title and abstract of an arXiv paper"
    hf = {key: (val, hf_description_type) for key, val in hf.items()}
    arx = {key: (val, arx_description_type) for key, val in arx.items()}



    for dataset in [hf, arx]:
        output = call_inference(
                schema = PathSchema,
                parsed_paper_text = dataset,
                graph_traversers = Traverser(CCS_HIERARCHY),
                return_dict_with_context = False,
                traversal_problem_type = "text2graph",
                # kwargs
                save = True,
                load = True,
                context_shortener = "full_paper",
                #ff_model = "4om",
                ff_model = "41n",
                #ff_model = "5n",
                )

def call_news_run(n=1):
    from metadata_schemas.acm_ccs_v3 import Traverser, CCS_HIERARCHY, v3_Schema, get_v3_traverser_dict
    from dataset_loader import Arxiv_HF_Newsletters_datasets
    ahd = Arxiv_HF_Newsletters_datasets()
    ahd.prepare()

    hf, arx, newsletters = ahd.get_dict_format(n)

    hf_description_type = "tags and model card of a Huggingface model"
    arx_description_type = "title and abstract of an arXiv paper"
    nl_description_type = "Newsletter item/blurb"

    hf = {key: (val, hf_description_type) for key, val in hf.items()}
    #arx = {key: (val, arx_description_type) for key, val in arx.items()}

    nls = {}
    for nl in newsletters:
        nls.update({key: (val, nl_description_type) for key, val in nl.items()})


    traversers = get_v3_traverser_dict() 
    for dataset in [nls]: # simply add hf and arx if doing all
        output = call_inference(
                schema = v3_Schema,
                parsed_paper_text = dataset,
                graph_traversers = traversers,
                return_dict_with_context = False,
                traversal_problem_type = "text2graph",
                # kwargs
                save = True,
                load = True,
                context_shortener = "full_paper",
                #ff_model = "4om",
                ff_model = "41n",
                #ff_model = "5n",
                )

def call_ccsv3_run(n=1):
    from metadata_schemas.acm_ccs_v3 import Traverser, CCS_HIERARCHY, v3_Schema, get_v3_traverser_dict
    from dataset_loader import Arxiv_HF_datasets
    ahd = Arxiv_HF_datasets()
    ahd.prepare()

    hf, arx = ahd.get_dict_format(n)

    hf_description_type = "Huggingface model, described by tags and model card"
    arx_description_type = "Arxiv paper, described by title and abstract"
    hf = {key: (val, hf_description_type) for key, val in hf.items()}
    arx = {key: (val, arx_description_type) for key, val in arx.items()}



    traversers = get_v3_traverser_dict() 
    for dataset in [
            hf, 
            arx
            ]:
        output = call_inference(
                schema = v3_Schema,
                parsed_paper_text = dataset,
                graph_traversers = traversers,
                return_dict_with_context = False,
                traversal_problem_type = "text2graph",
                # kwargs
                save = True,
                load = True,
                context_shortener = "full_paper",
                #ff_model = "4om",
                ff_model = "41n",
                #ff_model = "5n",
                )

    
def test_call():

    path = "/mnt/data/upcast/data/all_xmls/12093373_ascii_pmcoa.xml"
    path2= "/mnt/data/upcast/data/all_xmls/12095422_ascii_pmcoa.xml"
    import dataset_loader

    parsed_xml_paper_text = dataset_loader.load_paper_text_from_file_path(path)
    with open(path, "r") as f:
        raw_xml_paper_text = f.read()

    #paper_path = path
    paper_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/12093373/ascii"
    paper_url2= "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/12095422/ascii"


    from metadata_schemas.arxpr2_schema import Metadata_form as schema
    from metadata_schemas.arxpr2_schema import get_shuffled_form_generator
    schema = get_shuffled_form_generator(25, v3=True)()
    #from metadata_schemas.nhrf_qa_schema import Metadata_form as schema

    output = call_inference(
            schema,
            #
            # choose one to try out:
            #
            parsed_paper_text = parsed_xml_paper_text,#[1500:2000],
            #raw_xml_paper_text = raw_xml_paper_text,
            #paper_path = paper_path,
            #paper_path = [paper_path, path2],
            #paper_url = paper_url,
            #paper_url = {"paper1": paper_url, "paper2":paper_url2},
            #
            similarity_k = 2,
            field_info_to_compare = "choices",
            #field_info_to_compare = "description",
            context_shortener = "full_paper",
            #ff_model = "jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4",
            #ff_model = "llama3.1I-8b-q4",
            #ff_model = "TheBloke/Mistral-7B-v0.1-GPTQ",
            ff_model = "41n",
            )

    print("output:")
    import pprint
    pprint.pprint(output)

if __name__ == "__main__":
    #test_call()
    #call_ccsv3_run(5)
    call_ccsv3_run(10)
    call_news_run(10)
