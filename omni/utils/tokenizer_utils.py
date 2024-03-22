from dataclasses import dataclass, field

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils import is_tokenizers_available

from omni.utils.loguru import logger
from omni.utils.misc import deprecated

if is_tokenizers_available():
    from tokenizers import AddedToken
    from tokenizers import Encoding as EncodingFast
else:

    @dataclass(frozen=True, eq=True)
    class AddedToken:
        """
        AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
        way it should behave.
        """

        content: str = field(default_factory=str)
        single_word: bool = False
        lstrip: bool = False
        rstrip: bool = False
        normalized: bool = True

        def __getstate__(self):
            return self.__dict__

    @dataclass
    class EncodingFast:
        """This is dummy class because without the `tokenizers` library we don't have these objects anyway"""

        pass


@deprecated("average is not a good way to init the embedding, use `model.resize_token_embeddings` instead")
def smart_add_special_tokens(
    special_tokens_dict: dict[str, str | AddedToken],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """
    It is an extension of a tokenizer.add_special_tokens() function,
    adding the functionality of initializing parameters on top of it and automatically resize the token embedding

    Args:
        special_tokens_dict (Dict): Keys should be in the list of predefined special attributes:
            [`bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens`]
        tokenizer (PreTrainedTokenizer): _description_
        model (PreTrainedModel): _description_
    """
    # FIXME: This is the unoptimized version that may make your embedding size not be divisible by 64.
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    # resize input embeddings and output embeddings
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def average_init_token_embeddings(model: PreTrainedModel, num_added_tokens: int):
    assert num_added_tokens > 0, "`num_added_tokens` should be positive"
    logger.info("When initializing **newly added tokens** with average of existing embeddings, they must be **trainable**!")
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_added_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_added_tokens].mean(dim=0, keepdim=True)

    input_embeddings[-num_added_tokens:] = input_embeddings_avg
    output_embeddings[-num_added_tokens:] = output_embeddings_avg
