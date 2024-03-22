import copy
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import NewType, TypeAlias

from PIL import Image


# fmt: off
class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()

Role = NewType("Role", str)

@dataclass
class InterleavedContent:
    text: str
    text_list: list[str] = field(default_factory=list)
    image: list[Image.Image] = field(default_factory=list)
    image_placeholder: str = field(default=None)

    def __post_init__(self):
        if len(self.image) > 0:
            assert self.image_placeholder is not None, "image_placeholder must be specified when the number of self.image is greater than 0."
            self.text_list = self.text.split(self.image_placeholder)
            assert len(self.text_list) == len(self.image) + 1, f"The number of images `{len(self.image)}` and the number of image placeholders `{len(self.text_list) - 1}` should be the same."
        else:
            if self.image_placeholder is not None:
                assert self.image_placeholder not in self.text, f"image_placeholder `{self.image_placeholder}` is in text `{self.text}` but self.image is empty."

@dataclass
class Message:
    role: Role
    content: InterleavedContent

Dialog: TypeAlias = list[Message]
# fmt: on


@dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: list[Role]
    dialog: Dialog
    image_placeholder: str = field(default=None)
    image_list: list[Image.Image] = field(default_factory=list)
    sep_style: SeparatorStyle = field(default=SeparatorStyle.SINGLE)
    sep: list[str] = field(default_factory=lambda: ["###"])
    offset: int = field(default=0)
    version: str = field(default="Unknown")

    def __post_init__(self):
        users_list = ["user", "human"]
        assistants_list = ["assistant", "gpt", "bot"]

        assert any([user.lower() in self.roles[0].lower() for user in users_list]), f"self.role[0] must be user, but got {self.role[0]}"
        assert any([assistant.lower() in self.roles[1].lower() for assistant in assistants_list]), f"self.role[1] must be assistant, but got {self.role[1]}"

    def set_image_placeholder(self, image_placeholder: str):
        self.image_placeholder = image_placeholder

    def append_message(self, message: Message):
        role = message.role
        content = message.content

        assert role in self.roles, f"role {role} is not in {self.roles}"

        if content is not None:
            if content.image_placeholder is not None:
                assert (
                    self.image_placeholder == content.image_placeholder
                ), f"content.image_placeholder {content.image_placeholder} is not the same as self.image_placeholder {self.image_placeholder}"
            self.image_list += content.image

        self.dialog.append(message)

    def get_prompt(self):
        match self.sep_style:
            case SeparatorStyle.SINGLE | SeparatorStyle.TWO:
                sep_len = len(self.sep)
                ret = self.system + self.sep[0]
                for i, message in enumerate(self.dialog):
                    if message.content is not None:
                        ret += message.role + ": " + message.content.text + self.sep[i % sep_len]
                    else:
                        ret += message.role + ":"
                return ret

            case SeparatorStyle.MPT:
                ret = self.system + self.sep[0]
                for i, message in enumerate(self.dialog):
                    ret += message.role + message.content + self.sep[0]
                else:
                    ret += message.role
                return ret

            case _:
                raise ValueError(f"Invalid style: {self.sep_style}")

    def copy(self):
        return copy.deepcopy(self)

    def dict(self):
        return asdict(self)


conv_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=["Human", "Assistant"],
    dialog=[
        Message(role="Human", content=InterleavedContent(text="Give three tips for staying healthy.")),
        Message(
            role="Assistant",
            content=InterleavedContent(
                text="Sure, here are three tips for staying healthy:\n"
                "1. Exercise regularly: Regular physical activity can help improve your overall health and wellbeing. "
                "It can also help reduce your risk of chronic conditions such as obesity, diabetes, heart disease, "
                "and certain cancers. Aim for at least 150 minutes of moderate-intensity aerobic exercise or "
                "75 minutes of vigorous-intensity aerobic exercise per week, along with muscle-strengthening "
                "activities at least two days per week.\n"
                "2. Eat a balanced diet: Eating a balanced diet that is rich in fruits, "
                "vegetables, whole grains, lean proteins, and healthy fats can help support "
                "your overall health. Try to limit your intake of processed and high-sugar foods, "
                "and aim to drink plenty of water throughout the day.\n"
                "3. Get enough sleep: Getting enough quality sleep is essential for your physical "
                "and mental health. Adults should aim for seven to nine hours of sleep per night. "
                "Establish a regular sleep schedule and try to create a relaxing bedtime routine to "
                "help improve the quality of your sleep.",
            ),
        ),
    ],
    sep_style=SeparatorStyle.SINGLE,
    sep=["###"],
    offset=2,
)

conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=["Human", "Assistant"],
    dialog=[
        Message(role="Human", content=InterleavedContent(text="What are the key differences between renewable and non-renewable energy sources?")),
        Message(
            role="Assistant",
            content=InterleavedContent(
                text="Renewable energy sources are those that can be replenished naturally in a relatively "
                "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
                "Non-renewable energy sources, on the other hand, are finite and will eventually be "
                "depleted, such as coal, oil, and natural gas. Here are some key differences between "
                "renewable and non-renewable energy sources:\n"
                "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
                "energy sources are finite and will eventually run out.\n"
                "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
                "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
                "and other negative effects.\n"
                "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
                "have lower operational costs than non-renewable sources.\n"
                "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
                "locations than non-renewable sources.\n"
                "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
                "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
                "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
                "non-renewable sources are not, and their depletion can lead to economic and social instability.\n",
            ),
        ),
    ],
    sep_style=SeparatorStyle.SINGLE,
    sep=["###"],
    offset=2,
)

conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=["USER", "ASSISTANT"],
    dialog=[],
    sep_style=SeparatorStyle.TWO,
    sep=[" ", "</s>"],
    offset=0,
    version="v1",
)

conv_mpt = Conversation(
    system="<|im_start|>system\n"
    "- You are a helpful language and vision assistant.\n"
    "- You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n"
    "- You should follow the instructions carefully and explain your answers in detail.",
    roles=["<|im_start|>user\n", "<|im_start|>assistant\n"],
    dialog=[],
    sep_style=SeparatorStyle.MPT,
    sep=["<|im_end|>"],
    offset=0,
    version="mpt",
)

conv_mpt_text = Conversation(
    system="<|im_start|>system\n"
    "- You are a helpful assistant chatbot trained by MosaicML.\n"
    "- You answer questions.\n"
    "- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.\n"
    "- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.",
    roles=["<|im_start|>user\n", "<|im_start|>assistant\n"],
    dialog=[],
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
    offset=0,
    version="mpt",
)

# conv_bair_v1 = Conversation(
#     system="BEGINNING OF CONVERSATION:",
#     roles=("USER", "GPT"),
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.TWO,
#     sep=" ",
#     sep2="</s>",
# )

# simple_conv = Conversation(
#     system="A chat between a curious human and an artificial intelligence assistant. "
#     "The assistant gives helpful, detailed, and polite answers to the human's questions.",
#     roles=("Human", "Assistant"),
#     messages=(("Human", "Hi!"), ("Assistant", "Hi there! How can I help you today?")),
#     offset=2,
#     sep_style=SeparatorStyle.SINGLE,
#     sep="###",
# )

# simple_conv_multimodal = Conversation(
#     system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
#     "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
#     "Follow the instructions carefully and explain your answers in detail.",
#     roles=("Human", "Assistant"),
#     messages=(("Human", "Hi!"), ("Assistant", "Hi there!  How can I help you today?\n")),
#     offset=2,
#     sep_style=SeparatorStyle.SINGLE,
#     sep="###",
# )

# simple_conv_mpt_multimodal = Conversation(
#     system="""<|im_start|>system
# - You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab.
# - You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
# - You should follow the instructions carefully and explain your answers in detail.""",
#     roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
#     version="mpt",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.MPT,
#     sep="<|im_end|>",
# )

# simple_conv_legacy = Conversation(
#     system="You are LLaVA, a large language model trained by UW Madison WAIV Lab."
#     "You are designed to assist human with a variety of tasks using natural language."
#     "Follow the instructions carefully.",
#     roles=("Human", "Assistant"),
#     messages=(("Human", "Hi!\n\n### Response:"), ("Assistant", "Hi there!  How can I help you today?\n")),
#     offset=2,
#     sep_style=SeparatorStyle.SINGLE,
#     sep="###",
# )

# conv_llava_v1 = Conversation(
#     system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
#     "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
#     "Follow the instructions carefully and explain your answers in detail.",
#     roles=("USER", "ASSISTANT"),
#     version="v1",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.TWO,
#     sep=" ",
#     sep2="</s>",
# )

# dream_conversation = Conversation(
#     system=" A chat between a curious user and an artificial intelligence assistant. "
#     # "The assistant gives helpful, detailed, and polite answers to the user's questions. "
#     "The assistant writes engaged, and detailed articles/news/blogs/diaries/handbooks/menus/introductions "
#     "with illustrations and text following user's instructions.",
#     roles=("USER", "ASSISTANT"),
#     messages=(
#         (
#             "USER",
#             "Title: \"Sweet 'n' Spicy Sriracha Wings: A Flavorful Thai-Inspired Recipe\"\n\nInstructions:\n"
#             "1. Write a blog post about the preparation and cooking process of Sweet 'n' Spicy Sriracha Wings, inspired by the provided text.\n"
#             "2. Share some background information on Sriracha sauce, its Thai origin, and its versatility in different cuisines.\n"
#             "3. Break down the recipe and cooking process into clear, easy-to-follow steps, including ingredients, marination, cooking method, and sauce preparation.\n"
#             "4. Include a brief personal story or experience related to the recipe, such as Instagram engagement or marination in a bowl instead of a bag.\n"
#             "5. Length: 600-800 words.\n"
#             "6. Keywords: Sweet 'n' Spicy Sriracha Wings, Thai recipe, Sriracha sauce, chicken wings, marinade, Instagram, sauce preparation.\n"
#             "7. Category: Food and Cooking, Recipes, Thai Cuisine.",
#         ),
#         (
#             "ASSISTANT",
#             "Food Network Chef Tyler Florence. <dream_start> <dream_end>  "
#             "Photo by John Lee. "
#             "Is a meal even a meal in Ireland if there aren't a few potatoes thrown in somewhere? "
#             "Given our enthusiasm for the humble potato, it would be easy to assume that we know a thing or two about cooking them. "
#             "However, according to American Food Network chef Tyler Florence, you've probably been going wrong. "
#             "Most of us know the secret to decent mash is butter and milk heaped in once you've boiled and drained your potatoes, but according to Tyler, this is where we're all going wrong. <dream_start> <dream_end>  "
#             '"Potatoes have a really delicate, beautiful minerality to their flavour profile, and when you cook potatoes in water and you pour that water down the drain, you\'ve extracted all the flavour of the potato," he told PopSugar. '
#             '"So what I do with mashed potatoes \u2014 because you finish them with cream and butter anyway, right? '
#             "\u2014 is I'll take that same cream and butter and add that to the potatoes and cook the potatoes in cream and butter.\" "
#             "As well as the cream and butter, he also adds some olive oil, garlic, sage, rosemary and a little bit of thyme to his cold, peeled, chopped potatoes. "
#             "Once boiled and tender, Tyler puts a bowl on his counter, and a colander inside that bowl. "
#             '"I pour the potatoes through the colander, and the cream will collect on the bottom. '
#             "Then I'll put the potatoes back into the pot and take the potato-infused garlic cream and fold that back into the potatoes, and it's the most incredible flavour profile. "
#             "It's the best mashed potato you'll ever taste in your entire life.\" "
#             "Sounds delicious, right? "
#             "We'll definitely be giving it a whirl.",
#         ),
#     ),
#     version="v1",
#     offset=0,
#     sep_style=SeparatorStyle.TWO,
#     sep=" ",
#     sep2="</s>",
# # )

default_conversation = conv_vicuna_v1_1
conv_templates: dict[str, Conversation] = {
    "default_v1": conv_vicuna_v1_1,
    "default": conv_v1_2,
    # "simple": simple_conv,
    # "simple_legacy": simple_conv_legacy,
    # "multimodal": simple_conv_multimodal,
    # "dream_conv": dream_conversation,
    # "mpt_multimodal": simple_conv_mpt_multimodal,
    # "llava_v1": conv_llava_v1,
    # fastchat
    "v1": conv_v1_2,
    # "bair_v1": conv_bair_v1,
    "vicuna_v1_1": conv_vicuna_v1_1,
    "mpt": conv_mpt,
    "mpt_text": conv_mpt_text,
}


if __name__ == "__main__":
    print(conv_v1.get_prompt())
