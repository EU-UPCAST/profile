import argparse
import yaml
from types import SimpleNamespace

from load_modules import load_modules
from run_modules import FormFillingIterator




def add_defaults(parameters):
    """ given parameters for a run, add default values for all fields that are not included (from arguments.yaml) """
    with open("arguments.yaml", "r") as f:
        argument_template = yaml.safe_load(f)

    for argtype in argument_template:
        for argname in argument_template[argtype]:
            if not argname in parameters:
                parameters[argname] = argument_template[argtype][argname]["default"]
    return parameters

def call_inference(paper_text, schema, **kwargs):
    parameters = add_defaults(kwargs)

    args=SimpleNamespace(**parameters)

    args.load=False
    args.save=False
    args.schema = schema

    prepared_kwargs = load_modules(args)

    return FormFillingIterator(args, **prepared_kwargs).fill_single_form(key="", paper_text=paper_text, pydantic_form=schema, return_dict_with_context=True)



if __name__ == "__main__":
    # test it out:

    path = "/mnt/data/upcast/data/all_xmls/12093373_ascii_pmcoa.xml"
    import dataset_loader
    paper_text = dataset_loader.load_paper_text_from_file_path(path)

    from metadata_schemas.arxpr2_schema import Metadata_form as schema

    output = call_inference(paper_text, schema, 
            similarity_k = 5,
            field_info_to_compare = "choices",
            )

    print("output:")
    import pprint
    pprint.pprint(output)

