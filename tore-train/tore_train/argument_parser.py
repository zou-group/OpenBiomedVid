import dataclasses
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)

def combine_arguments(*args_dataclasses: Union[dataclass, List[dataclass]]):
    if len(args_dataclasses) == 1 and isinstance(args_dataclasses[0], list):
        args_dataclasses = args_dataclasses[0]
    fields = {}
    for args_dataclass in args_dataclasses:
        for field in dataclasses.fields(args_dataclass):
            if field.name not in fields:
                fields[field.name] = (field.type, field)
    return dataclasses.make_dataclass("CombinedArguments", [(name, type_, field) for name, (type_, field) in fields.items()])


class ToreTrainArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(
        self,
        yaml_arg: str,
        other_args: Optional[List[str]] = None,
        allow_extra_keys: bool = False,
    ) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg), allow_extra_keys=allow_extra_keys)

        outputs = []
        # Parse other_args into a dictionary supporting both "--arg=val" and "--arg val"
        other_args_dict = {}
        args_iter = iter(other_args or [])
        for arg in args_iter:
            if arg.startswith("--"):
                arg_name = arg.lstrip("-")
                if "=" in arg_name:
                    key, value = arg_name.split("=", 1)
                    other_args_dict[key] = value
                else:
                    try:
                        # Peek at the next argument to see if it's a value or another flag
                        next_arg = next(args_iter)
                        if next_arg.startswith("--"):
                            # It's another flag; treat the current arg as a boolean flag
                            other_args_dict[arg_name] = True
                            # Re-process the next_arg in the next iteration
                            args_iter = self._prepend_iterator(next_arg, args_iter)
                        else:
                            # It's a value for the current arg
                            other_args_dict[arg_name] = next_arg
                    except StopIteration:
                        # No more arguments; treat the current arg as a boolean flag
                        other_args_dict[arg_name] = True

        used_args = {}

        # Overwrite the default/loaded value with the value provided to the command line
        # Adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args_dict.items():
                # Add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # Cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    if "typing.Dict" in str(base_type) or "typing.List" in str(base_type):
                        inputs[arg] = json.loads(val)

                    # Bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if isinstance(val, bool):
                            inputs[arg] = val
                        elif isinstance(val, str):
                            if val.lower() in ["true", "1", "yes"]:
                                inputs[arg] = True
                            elif val.lower() in ["false", "0", "no"]:
                                inputs[arg] = False
                            else:
                                raise ValueError(f"Invalid boolean value: {val}")
                        else:
                            raise ValueError(f"Invalid boolean value type: {type(val)}")

                    # Add to used_args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            # We only do post_init once
            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def _prepend_iterator(self, value, iterator):
        """
        Helper function to prepend a single value back to an iterator.
        """
        yield value
        yield from iterator

    def parse(self) -> Union[DataClassType, Tuple[DataClassType]]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(
                os.path.abspath(sys.argv[1]), sys.argv[2:]
            )
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output