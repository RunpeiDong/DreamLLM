import copy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TypeAlias

from omni.conversation.multimodal import MultimodalContent


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_TWO = auto()
    LLAMA2 = auto()


@dataclass
class Message:
    role: str
    content: MultimodalContent | None = field(
        default=None, metadata={"help": "The content of the message can be pure text or multimodal content."}
    )


Dialog: TypeAlias = list[Message]


@dataclass
class Conversation:
    """
    A class that manages prompt templates and keeps all conversation history.
    """

    name: str = field(default="", metadata={"help": "The name of this template."})
    system_template: str = field(default="{system_message}", metadata={"help": "The template of the system prompt."})
    system_message: str = field(default="", metadata={"help": "The system message."})
    roles: tuple[str, str] = field(default=("USER", "ASSISTANT"), metadata={"help": "The names of two roles."})
    dialog: Dialog = field(
        default_factory=list,
        metadata={"help": "The dialog history. Each item is Message(role: str, content: str | MultimodalContent)."},
    )
    sep_style: SeparatorStyle = field(default=SeparatorStyle.ADD_COLON_TWO, metadata={"help": "The separator style."})
    sep: str = field(default="\n", metadata={"help": "The separator."})
    sep2: str | None = field(default=None, metadata={"help": "The second separator."})

    def get_prompt(self):
        system_prompt = self.system_template.format(system_message=self.system_message)
        match self.sep_style:
            case SeparatorStyle.ADD_COLON_TWO:
                seps = [self.sep, self.sep2]
                ret = system_prompt + seps[0]
                for i, message in enumerate(self.dialog):
                    if message.content is not None:
                        ret += message.role + ": " + message.content.text + seps[i % 2]
                    else:
                        ret += message.role + ":"
                return ret

            case SeparatorStyle.LLAMA2:
                seps = [self.sep, self.sep2]
                if self.system_message:
                    ret = system_prompt
                else:
                    ret = "[INST] "
                for i, message in enumerate(self.dialog):
                    tag = self.roles[i % 2]
                    if message.content is not None:
                        if i == 0:
                            ret += message.content.text + " "
                        else:
                            ret += tag + " " + message.content.text + seps[i % 2]
                    else:
                        ret += tag
                return ret

            case _:
                raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, message: Message):
        """Append a new message."""
        self.dialog.append(message)

    def update_last_message(self, message: Message):
        """
        Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.dialog[-1] = message

    def reset_dialog(self, dialog: Dialog):
        self.dialog: Dialog = []
        for i, message in enumerate(dialog):
            message.role = self.roles[i % 2]
            self.append_message(message)

    def copy(self):
        return copy.deepcopy(self)


# A global registry for all conversation templates
conv_templates: dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# vicuna1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# llama2 template
# reference: https://huggingface.co/blog/codellama#conversational-instructions
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)
