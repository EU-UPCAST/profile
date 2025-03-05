import argparse
import yaml
from types import SimpleNamespace

from load_modules import load_modules
from run_modules import FormFillingIterator


def parse_terminal_arguments():

    import yaml
    with open("arguments.yaml", "r") as f:
        argument_template = yaml.safe_load(f)

    parser = argparse.ArgumentParser()

    parser.add_argument("paper_path", help = "path to xml file (or possibly folder of xmls in future)")
    parser.add_argument("schema_path", help = "path to pydantic schema")
    parser.add_argument("output_path", help = "path to output json")

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


def run_inference(args):

    args.dataset = None
    args.load=False
    args.save=False

    prepared_kwargs = load_modules(args)

    FormFillingIterator(args, **prepared_kwargs)()


def add_defaults(parameters):
    """ given parameters for a run, add default values for all fields that are not included (from arguments.yaml) """
    with open("arguments.yaml", "r") as f:
        argument_template = yaml.safe_load(f)

    for argtype in argument_template:
        for argname in argument_template[argtype]:
            if not argname in parameters:
                parameters[argname] = argument_template[argtype][argname]["default"]
    return parameters


def call_inference_from_func(paper_path, schema_path, output_path, **kwargs):
    # TODO: take file objects instead of paths
    parameters = add_defaults(kwargs)
    parameters["paper_path"]=paper_path
    parameters["schema_path"]=schema_path
    parameters["output_path"]=output_path
    args=SimpleNamespace(**parameters)
    run_inference(args)
    # TODO: return json_obj,

def call_inference_from_terminal():
    args = parse_terminal_arguments()
    run_inference(args)

if __name__ == "__main__":
    call_inference_from_terminal()
    #call_inference_from_func(
    #        "/mnt/data/upcast/data/all_xmls/12093373_ascii_pmcoa.xml",
    #        "metadata_schemas/arxpr2_schema.py",
    #        "inference_output.json",
    #        similarity_k = 5,
    #        field_info_to_compare = "choices",
    #        )
