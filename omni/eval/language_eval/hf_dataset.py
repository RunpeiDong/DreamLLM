import re
import torch
from datasets import load_dataset


DATASET_NAME = {
    "boolq": ["boolq"],
    "piqa": ["piqa"],
    "siqa": ["social_i_qa"],
    "hellaswag": ["hellaswag"],
    "winogrande": ["winogrande", "winogrande_xl"],
    "arc_e": ["ai2_arc", "ARC-Easy"],
    "arc_c": ["ai2_arc", "ARC-Challenge"],
    "obqa": ["openbookqa", "main"],
    "race_m": ["race", "middle"],
    "race_h": ["race", "high"],
    "triviaqa": ["trivia_qa", "rc"],
    "naturalqa": ["nq_open"],
    "drop_gen": ["drop"],
    "clue_c3": ["clue", "c3"],
    "clue_wsc": ["clue", "cluewsc2020"],
    "clue_cmrc": ["clue", "cmrc2018"],
}


CHOICES = ["A", "B", "C", "D", "E", "F"]


class LLaMa_Multich_Dataset:
    labels = CHOICES[:4]
    forward_once = True

    def __init__(self, dataset, ntrain, lite=False):
        dataset = dataset.lower()
        self.test_df = load_dataset(*DATASET_NAME[dataset], split="validation")
        if lite and len(self.test_df) > 500:
            self.test_df = load_dataset(*DATASET_NAME[dataset], split="validation[:500]")
        self.train_prompt = self.gen_train_prompts(dataset, ntrain)

        self.ntrain = ntrain

    def gen_train_prompts(self, dataset, ntrain):
        dev_df = load_dataset(*DATASET_NAME[dataset], split="train")
        train_prompt = ""
        for idx, sample in enumerate(dev_df):
            if idx == ntrain:
                break
            train_prompt = train_prompt + self.make_train_prompt(sample)
        return train_prompt

    def make_test_prompt(self, sample):
        raise NotImplementedError

    def get_label(self, sample):
        raise NotImplementedError

    def make_train_prompt(self, sample):
        label = self.get_label(sample)
        return self.make_test_prompt(sample) + f"{CHOICES[label]}\n\n"

    def __getitem__(self, idx):
        question = self.test_df[idx]
        prompt = self.make_test_prompt(question)
        prompt = self.train_prompt + prompt
        label = self.get_label(question)
        return prompt, label

    def __call__(self, idx):
        return self.__getitem__(idx)

    def __len__(self):
        return self.test_df.num_rows


class Generation_Dataset(LLaMa_Multich_Dataset):
    forward_once = False
    prefix = ""
    len_label = 1

    def __getitem__(self, idx):
        sample = self.test_df[idx]
        question = self.make_test_prompt(sample)
        labels = self.get_label(sample)
        question = self.prefix + self.train_prompt + question
        return question, labels

    def make_train_prompt(self, sample):
        labels = self.get_label(sample)
        return self.make_test_prompt(sample) + f" {labels[0]}\n\n"


class BoolqDataset(LLaMa_Multich_Dataset):
    labels = ["no", "yes"]

    def __init__(self, *args, **kwargs):
        super().__init__("boolq", *args, **kwargs)

    def make_test_prompt(self, sample):
        return "Choose yes or no to answer the question.\n\n" + f'{sample["passage"]}\n\n{sample["question"][0].upper()+sample["question"][1:]}?'

    def make_train_prompt(self, sample):
        return self.make_test_prompt(sample) + f'{"yes" if sample["answer"] is True else "no"}\n\n'

    def get_label(self, sample):
        return 1 if sample["answer"] is True else 0


class PIQADataset(LLaMa_Multich_Dataset):
    labels = CHOICES[:2]

    def __init__(self, *args, **kwargs):
        super().__init__("piqa", *args, **kwargs)

    def make_test_prompt(self, sample):
        return f'Choose A or B to answer the question.\n\n{sample["goal"]}\n\nOptions:\nA. {sample["sol1"]}\nB. {sample["sol2"]}'

    def get_label(self, sample):
        return sample["label"]


class SIQADataset(LLaMa_Multich_Dataset):
    labels = CHOICES[:3]

    def __init__(self, *args, **kwargs):
        super().__init__("siqa", *args, **kwargs)

    def make_test_prompt(self, sample):
        s = f'Choose A, B or C to answer the question.\n\n{sample["context"]}\n\n{sample["question"]}\n\n'
        s += "Options:\n"
        s += f'A. {sample["answerA"]}\n'
        s += f'B. {sample["answerB"]}\n'
        s += f'C. {sample["answerC"]}'
        return s

    def get_label(self, sample):
        return int(sample["label"]) - 1


class HellaSwagDataset(LLaMa_Multich_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__("hellaswag", *args, **kwargs)

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        text = text.replace("..", ".")
        return text.strip()

    def make_test_prompt(self, sample):
        s = f'Choose A, B, C or D to complete sentence about {sample["activity_label"]}.\n\n'
        s += sample["ctx_a"] + " " + sample["ctx_b"].capitalize()
        s += "\n\nOptions:\n"
        s += f'A. {sample["endings"][0]}\n'
        s += f'B. {sample["endings"][1]}\n'
        s += f'C. {sample["endings"][2]}\n'
        s += f'D. {sample["endings"][3]}'
        return self.preprocess(s)

    def get_label(self, sample):
        return int(sample["label"])


class WinoGrandeDataset(LLaMa_Multich_Dataset):
    labels = CHOICES[:2]

    def __init__(self, *args, **kwargs):
        super().__init__("winogrande", *args, **kwargs)

    def make_test_prompt(self, sample):
        s = f'Choose A ot B to complete the sentence.\n\n{sample["sentence"]}\n\n'
        s += f"Options:\n"
        s += f'A. {sample["option1"]}\n'
        s += f'B. {sample["option2"]}'
        return s

    def get_label(self, sample):
        return int(sample["answer"]) - 1


class ARC_EDataset(LLaMa_Multich_Dataset):
    labels = CHOICES[:5]

    def __init__(self, *args, **kwargs):
        super().__init__("arc_e", *args, **kwargs)

    def make_test_prompt(self, sample):
        choices = sample["choices"]
        s = f'Choose A, B, C, D or E to answer the question. \n\n{sample["question"]}\n\n'  # \n'
        for i in range(len(choices["label"])):
            s += f'{CHOICES[i]}. {choices["text"][i]}\n'  # \n'
        return s.strip("\n")

    def get_label(self, sample):
        choices = sample["choices"]
        label = choices["label"].index(sample["answerKey"])
        return label


class ARC_CDataset(ARC_EDataset):
    def __init__(self, *args, **kwargs):
        super(ARC_EDataset, self).__init__("arc_c", *args, **kwargs)


class OBQADataset(LLaMa_Multich_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__("obqa", *args, **kwargs)

    def make_test_prompt(self, sample):
        s = f'Choose A, B, C or D to complete the sentence.\n\n{sample["question_stem"]}\n\n'
        s += "Options:\n"
        choices = sample["choices"]
        for i in range(len(choices["label"])):
            s += f'{CHOICES[i]}. {choices["text"][i]}\n'
        return s.strip("\n")

    def get_label(self, sample):
        choices = sample["choices"]
        label = choices["label"].index(sample["answerKey"])
        return label


class RACEMiddleDataset(LLaMa_Multich_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__("race_m", *args, **kwargs)

    def make_test_prompt(self, sample):
        s = f'Choose {", ".join([CHOICES[i] for i in range(len(sample["options"]))])} to answer the question.\n\n{sample["article"]}\n\n{sample["question"]}\n\n'  # 46.54
        s = s.replace("  ", " ").replace("..", ".").strip()
        s += "Options:\n"
        for i in range(len(sample["options"])):
            s += f'{CHOICES[i]}. {sample["options"][i].replace("  ", " ").replace("..", ".").strip()}\n'
        return s.strip("\n")

    def get_label(self, sample):
        return CHOICES.index(sample["answer"])


class RACEHighDataset(RACEMiddleDataset):
    def __init__(self, *args, **kwargs):
        super(RACEMiddleDataset, self).__init__("race_h", *args, **kwargs)


class TriviaQADataset(Generation_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__("triviaqa", *args, **kwargs)

    def normalize_question(self, text):
        def white_space_fix(text):
            return " ".join(text.split())

        def handle_punc(text):
            exclude = set("".join(['"', "‘", "’", "´", "`"]))
            return "".join(ch if ch not in exclude else " " for ch in text)

        return white_space_fix(handle_punc(text)).strip().strip("?").strip() + "?"

    def make_test_prompt(self, sample):
        question = self.normalize_question(sample["question"])
        return question

    def get_label(self, sample):
        answers = sample["answer"]
        labels = [answers["value"]]
        return labels


class NaturalQuestionsDataset(Generation_Dataset):
    len_label = 23

    def __init__(self, *args, **kwargs):
        super().__init__("naturalqa", *args, **kwargs)

    def make_test_prompt(self, sample):
        prefix = "Give me an accurate answer in five words. \n\n"  # 14.5 20.3
        question = prefix + sample["question"].replace("?", "").strip().capitalize() + "?"
        return question

    def get_label(self, sample):
        labels = sample["answer"]
        return labels + ["NAN"] * (self.len_label - len(labels))


class DropDataset(Generation_Dataset):
    len_label = 19

    def __init__(self, *args, **kwargs):
        super().__init__("drop_gen", *args, **kwargs)

    def make_test_prompt(self, sample):
        question = sample["passage"].strip() + "\n\n" + sample["question"]
        return question

    def get_label(self, sample):
        labels = sample["answers_spans"]["spans"]
        return labels + ["NAN"] * (self.len_label - len(labels))


class CLUEC3Dataset(LLaMa_Multich_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__("clue_c3", *args, **kwargs)

    def make_test_prompt(self, sample):
        context = "\n".join(sample["context"])
        s = f'请用A、B、C、D根据以下对话回答问题。\n\n{context}\n\n{sample["question"]}\n\n选项：\n'
        for i in range(len(sample["choice"])):
            s += f'{CHOICES[i]}. {sample["choice"][i]}\n'
        return s.strip("\n")

    def get_label(self, sample):
        label = sample["choice"].index(sample["answer"])
        return label


class CLUEWSCDataset(LLaMa_Multich_Dataset):
    labels = ["正确", "错误"]

    def __init__(self, *args, **kwargs):
        super().__init__("clue_wsc", *args, **kwargs)

    def make_test_prompt(self, sample):
        question = "阅读这句话，判断指代关系是否正确。\n\n"
        question += sample["text"]
        question += f'\n\n{sample["target"]["span2_text"]}是否指代{sample["target"]["span1_text"]}？'
        return question

    def get_label(self, sample):
        label = sample["label"]
        return label


class CLUECMRCDataset(Generation_Dataset):
    prefix = ""
    len_label = 3

    def __init__(self, *args, **kwargs):
        super().__init__("clue_cmrc", *args, **kwargs)

    def make_test_prompt(self, sample):
        question = "根据下面这段话简短并且准确地回答问题。\n\n"
        question += sample["context"]
        question += f'\n\n问题：{sample["question"]}'
        return question

    def get_label(self, sample):
        labels = sample["answers"]["text"]
        return labels + ["NAN"] * (self.len_label - len(labels))


DATASET_CLASS = {
    "boolq": BoolqDataset,
    "piqa": PIQADataset,
    "siqa": SIQADataset,
    "hellaswag": HellaSwagDataset,
    "winogrande": WinoGrandeDataset,
    "arc_e": ARC_EDataset,
    "arc_c": ARC_CDataset,
    "obqa": OBQADataset,
    "race_m": RACEMiddleDataset,
    "race_h": RACEHighDataset,
    "triviaqa": TriviaQADataset,
    "naturalqa": NaturalQuestionsDataset,
    "drop_gen": DropDataset,
    "clue_c3": CLUEC3Dataset,
    "clue_wsc": CLUEWSCDataset,
    "clue_cmrc": CLUECMRCDataset,
}


def list_datasets():
    return DATASET_NAME.keys()


def load_dataset_by_name(dataset_name, *args, **kwargs):
    if dataset_name not in DATASET_CLASS.keys():
        raise NotImplementedError(f'{dataset_name} is not in [{", ".join(list_datasets())}].')
    return DATASET_CLASS[dataset_name](*args, **kwargs)
