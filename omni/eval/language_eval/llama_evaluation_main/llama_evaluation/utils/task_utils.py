import os
import numpy as np
from omni.utils.loguru import logger


__all__ = ["BASE_TASKS", "CEVAL_CATS", "MMLU_CATS", "CMMLU_CATS", "TASK_MAPPING", "dataset_mapping", "metrics_post_process"]

DATAROOT = os.environ.get('EVAL_DATASET_PATH', "data/datasets")
CEVAL_DATAPATH = os.path.join(DATAROOT, "ceval/")
MMLU_DATAPATH = os.path.join(DATAROOT, "mmlu/")
CMMLU_DATAPATH = os.path.join(DATAROOT, "cmmlu/")
AGI_DATAPATH = os.path.join(DATAROOT,"AGIEval/v1")
GAOKAO_DATAPATH = os.path.join(DATAROOT, "GAOKAO_bench")


CEVAL_CATS = {
        "accountant": ['Accountant', '注册会计师', 'Other', 49],
        "advanced_mathematics": ['Advanced Mathematics', '高等数学', 'STEM', 19],
        "art_studies": ['Art Studies', '艺术学', 'Humanities', 33],
        "basic_medicine": ['Basic Medicine', '基础医学', 'Other', 19],
        "business_administration": ['Business Administration', '工商管理', 'Social Science', 33],
        "chinese_language_and_literature": ['Chinese Language and Literature', '中国语言文学', 'Humanities', 23],
        "civil_servant": ['Civil Servant', '公务员', 'Other', 47],
        "clinical_medicine": ['Clinical Medicine', '临床医学', 'Other', 22],
        "college_chemistry": ['College Chemistry', '大学化学', 'STEM', 24],
        "college_economics": ['College Economics', '大学经济学', 'Social Science', 55],
        "college_physics": ['College Physics', '大学物理', 'STEM', 19],
        "college_programming": ['College Programming', '大学编程', 'STEM', 37],
        "computer_architecture": ['Computer Architecture', '计算机组成', 'STEM', 21],
        "computer_network": ['Computer Network', '计算机网络', 'STEM', 19],
        "discrete_mathematics": ['Discrete Mathematics', '离散数学', 'STEM', 16],
        "education_science": ['Education Science', '教育学', 'Social Science', 29],
        "electrical_engineer": ['Electrical Engineer', '注册电气工程师', 'STEM', 37],
        "environmental_impact_assessment_engineer": ['Environmental Impact Assessment Engineer', '环境影响评价工程师', 'Other', 31],
        "fire_engineer": ['Fire Engineer', '注册消防工程师', 'Other', 31],
        "high_school_biology": ['High School Biology', '高中生物', 'STEM', 19],
        "high_school_chemistry": ['High School Chemistry', '高中化学', 'STEM', 19],
        "high_school_chinese": ['High School Chinese', '高中语文', 'Humanities', 19],
        "high_school_geography": ['High School Geography', '高中地理', 'Social Science', 19],
        "high_school_history": ['High School History', '高中历史', 'Humanities', 20],
        "high_school_mathematics": ['High School Mathematics', '高中数学', 'STEM', 18],
        "high_school_physics": ['High School Physics', '高中物理', 'STEM', 19],
        "high_school_politics": ['High School Politics', '高中政治', 'Social Science', 19],
        "ideological_and_moral_cultivation": ['Ideological and Moral Cultivation', '思想道德修养与法律基础', 'Humanities', 19],
        "law": ['Law', '法学', 'Humanities', 24],
        "legal_professional": ['Legal Professional', '法律职业资格', 'Humanities', 23],
        "logic": ['Logic', '逻辑学', 'Humanities', 22],
        "mao_zedong_thought": ['Mao Zedong Thought', '毛泽东思想和中国特色社会主义理论体系概论', 'Social Science', 24],
        "marxism": ['Marxism', '马克思主义基本原理', 'Social Science', 19],
        "metrology_engineer": ['Metrology Engineer', '注册计量师', 'STEM', 24],
        "middle_school_biology": ['Middle School Biology', '初中生物', 'STEM', 21],
        "middle_school_chemistry": ['Middle School Chemistry', '初中化学', 'STEM', 20],
        "middle_school_geography": ['Middle School Geography', '初中地理', 'Social Science', 12],
        "middle_school_history": ['Middle School History', '初中历史', 'Humanities', 22],
        "middle_school_mathematics": ['Middle School Mathematics', '初中数学', 'STEM', 19],
        "middle_school_physics": ['Middle School Physics', '初中物理', 'STEM', 19],
        "middle_school_politics": ['Middle School Politics', '初中政治', 'Social Science', 21],
        "modern_chinese_history": ['Modern Chinese History', '近代史纲要', 'Humanities', 23],
        "operating_system": ['Operating System', '操作系统', 'STEM', 19],
        "physician": ['Physician', '医师资格', 'Other', 49],
        "plant_protection": ['Plant Protection', '植物保护', 'Other', 22],
        "probability_and_statistics": ['Probability and Statistics', '概率统计', 'STEM', 18],
        "professional_tour_guide": ['Professional Tour Guide', '导游资格', 'Humanities', 29],
        "sports_science": ['Sports Science', '体育学', 'Other', 19],
        "tax_accountant": ['Tax Accountant', '税务师', 'Other', 49],
        "teacher_qualification": ['Teacher Qualification', '教师资格', 'Social Science', 44],
        "urban_and_rural_planner": ['Urban and Rural Planner', '注册城乡规划师', 'Other', 46],
        "veterinary_medicine": ['Veterinary Medicine', '兽医学', 'STEM', 23],
}


MMLU_CATS = {
        'abstract_algebra': ['math', 'STEM', 99],
        'anatomy': ['health', 'other (business, health, misc.)', 134],
        'astronomy': ['physics', 'STEM', 151],
        'business_ethics': ['business', 'other (business, health, misc.)', 99],
        'clinical_knowledge': ['health', 'other (business, health, misc.)', 264],
        'college_biology': ['biology', 'STEM', 143],
        'college_chemistry': ['chemistry', 'STEM', 99],
        'college_computer_science': ['computer science', 'STEM', 99],
        'college_mathematics': ['math', 'STEM', 99],
        'college_medicine': ['health', 'other (business, health, misc.)', 172],
        'college_physics': ['physics', 'STEM', 101],
        'computer_security': ['computer science', 'STEM', 99],
        'conceptual_physics': ['physics', 'STEM', 234],
        'econometrics': ['economics', 'social sciences', 113],
        'electrical_engineering': ['engineering', 'STEM', 144],
        'elementary_mathematics': ['math', 'STEM', 377],
        'formal_logic': ['philosophy', 'humanities', 125],
        'global_facts': ['other', 'other (business, health, misc.)', 99],
        'high_school_biology': ['biology', 'STEM', 309],
        'high_school_chemistry': ['chemistry', 'STEM', 202],
        'high_school_computer_science': ['computer science', 'STEM', 99],
        'high_school_european_history': ['history', 'humanities', 164],
        'high_school_geography': ['geography', 'social sciences', 197],
        'high_school_government_and_politics': ['politics', 'social sciences', 192],
        'high_school_macroeconomics': ['economics', 'social sciences', 389],
        'high_school_mathematics': ['math', 'STEM', 269],
        'high_school_microeconomics': ['economics', 'social sciences', 237],
        'high_school_physics': ['physics', 'STEM', 150],
        'high_school_psychology': ['psychology', 'social sciences', 544],
        'high_school_statistics': ['math', 'STEM', 215],
        'high_school_us_history': ['history', 'humanities', 203],
        'high_school_world_history': ['history', 'humanities', 236],
        'human_aging': ['health', 'other (business, health, misc.)', 222],
        'human_sexuality': ['culture', 'social sciences', 130],
        'international_law': ['law', 'humanities', 120],
        'jurisprudence': ['law', 'humanities', 107],
        'logical_fallacies': ['philosophy', 'humanities', 162],
        'machine_learning': ['computer science', 'STEM', 111],
        'management': ['business', 'other (business, health, misc.)', 102],
        'marketing': ['business', 'other (business, health, misc.)', 233],
        'medical_genetics': ['health', 'other (business, health, misc.)', 99],
        'miscellaneous': ['other', 'other (business, health, misc.)', 782],
        'moral_disputes': ['philosophy', 'humanities', 345],
        'moral_scenarios': ['philosophy', 'humanities', 894],
        'nutrition': ['health', 'other (business, health, misc.)', 305],
        'philosophy': ['philosophy', 'humanities', 310],
        'prehistory': ['history', 'humanities', 323],
        'professional_accounting': ['other', 'other (business, health, misc.)', 281],
        'professional_law': ['law', 'humanities', 1533],
        'professional_medicine': ['health', 'other (business, health, misc.)', 271],
        'professional_psychology': ['psychology', 'social sciences', 611],
        'public_relations': ['politics', 'social sciences', 109],
        'security_studies': ['politics', 'social sciences', 244],
        'sociology': ['culture', 'social sciences', 200],
        'us_foreign_policy': ['politics', 'social sciences', 99],
        'virology': ['health', 'other (business, health, misc.)', 165],
        'world_religions': ['philosophy', 'humanities', 170]
}


CMMLU_CATS = {
        'agronomy': ['农学', 'other', 'Other', 169],
        'anatomy': ['解剖学', 'biology', 'STEM', 148],
        'ancient_chinese': ['古汉语', 'linguistics', 'Social Science', 164],
        'arts': ['艺术学', 'arts', 'Humanities', 160],
        'astronomy': ['天文学', 'physics', 'STEM', 165],
        'business_ethics': ['商业伦理', 'business', 'Social Science', 209],
        'chinese_civil_service_exam': ['中国公务员考试', 'politics', 'Social Science', 160],
        'chinese_driving_rule': ['中国驾驶规则', 'other', 'Other', 131],
        'chinese_food_culture': ['中国饮食文化', 'culture', 'Social Science', 136],
        'chinese_foreign_policy': ['中国外交政策', 'politics', 'Social Science', 107],
        'chinese_history': ['中国历史', 'history', 'Humanities', 323],
        'chinese_literature': ['中国文学', 'literature', 'Humanities', 204],
        'chinese_teacher_qualification': ['中国教师资格', 'education', 'Social Science', 179],
        'clinical_knowledge': ['临床知识', 'other', 'Other', 237],
        'college_actuarial_science': ['大学精算学', 'math', 'STEM', 106],
        'college_education': ['大学教育学', 'education', 'Social Science', 107],
        'college_engineering_hydrology': ['大学工程水文学', 'engineering', 'STEM', 106],
        'college_law': ['大学法律', 'law', 'Humanities', 108],
        'college_mathematics': ['大学数学', 'math', 'STEM', 105],
        'college_medical_statistics': ['大学医学统计', 'statistics', 'STEM', 106],
        'college_medicine': ['大学医学', 'other', 'Other', 273],
        'computer_security': ['计算机安全', 'other', 'Other', 171],
        'conceptual_physics': ['概念物理学', 'physics', 'STEM', 147],
        'construction_project_management': ['建设工程管理', 'other', 'Other', 139],
        'economics': ['经济学', 'economics', 'Social Science', 159],
        'education': ['教育学', 'education', 'Social Science', 163],
        'electrical_engineering': ['电气工程', 'engineering', 'STEM', 172],
        'elementary_chinese': ['小学语文', 'linguistics', 'Social Science', 252],
        'elementary_commonsense': ['小学常识', 'other', 'Other', 198],
        'elementary_information_and_technology': ['小学信息技术', 'other', 'Other', 238],
        'elementary_mathematics': ['初等数学', 'math', 'STEM', 230],
        'ethnology': ['民族学', 'culture', 'Social Science', 135],
        'food_science': ['食品科学', 'other', 'Other', 143],
        'genetics': ['遗传学', 'biology', 'STEM', 176],
        'global_facts': ['全球事实', 'global', 'Humanities', 149],
        'high_school_biology': ['高中生物', 'biology', 'STEM', 169],
        'high_school_chemistry': ['高中化学', 'chemistry', 'STEM', 132],
        'computer_science': ['计算机科学', 'computer science', 'STEM', 204],
        'high_school_geography': ['高中地理', 'geography', 'Social Science', 118],
        'high_school_mathematics': ['高中数学', 'math', 'STEM', 164],
        'high_school_physics': ['高中物理学', 'physics', 'STEM', 110],
        'high_school_politics': ['高中政治', 'politics', 'Social Science', 143],
        'human_sexuality': ['人类性行为', 'other', 'Other', 126],
        'international_law': ['国际法学', 'law', 'Humanities', 185],
        'journalism': ['新闻学', 'sociology', 'Social Science', 172],
        'jurisprudence': ['法理学', 'law', 'Humanities', 411],
        'legal_and_moral_basis': ['法律与道德基础', 'other', 'Other', 214],
        'logical': ['逻辑学', 'philosophy', 'Humanities', 123],
        'machine_learning': ['机器学习', 'computer science', 'STEM', 122],
        'management': ['管理学', 'business', 'Social Science', 210],
        'marketing': ['市场营销', 'business', 'Social Science', 180],
        'marxist_theory': ['马克思主义理论', 'philosophy', 'Humanities', 189],
        'modern_chinese': ['现代汉语', 'linguistics', 'Social Science', 116],
        'nutrition': ['营养学', 'other', 'Other', 145],
        'philosophy': ['哲学', 'philosophy', 'Humanities', 105],
        'professional_accounting': ['专业会计', 'business', 'Social Science', 175],
        'professional_law': ['专业法学', 'law', 'Humanities', 211],
        'professional_medicine': ['专业医学', 'other', 'Other', 376],
        'professional_psychology': ['专业心理学', 'psychology', 'Social Science', 232],
        'public_relations': ['公共关系', 'politics', 'Social Science', 174],
        'security_study': ['安全研究', 'politics', 'Social Science', 135],
        'sociology': ['社会学', 'culture', 'Social Science', 226],
        'sports_science': ['体育学', 'other', 'Other', 165],
        'traditional_chinese_medicine': ['中医中药', 'other', 'Other', 185],
        'virology': ['病毒学', 'biology', 'STEM', 169],
        'world_history': ['世界历史', 'history', 'Humanities', 161],
        'world_religions': ['世界宗教', 'global', 'Humanities', 160],
}

CMMLU_CHINESE_SPECIFIC = [
        'ancient_chinese',
        'chinese_civil_service_exam',
        'chinese_driving_rule',
        'chinese_food_culture',
        'chinese_foreign_policy',
        'chinese_history',
        'chinese_literature',
        'chinese_teacher_qualification',
        'construction_project_management',
        'elementary_chinese',
        'elementary_commonsense',
        'ethnology',
        'high_school_politics',
        'modern_chinese',
        'traditional_chinese_medicine'
]


GAOKAO_CATS = {
        "math1": ['2010-2022_Math_I_MCQs', 'single_choice', 214 * 5],
        "math2": ['2010-2022_Math_II_MCQs', 'single_choice', 218 * 5],
        "biology": ['2010-2022_Biology_MCQs', 'single_choice', 150 * 6],
        "chemistry": ['2010-2022_Chemistry_MCQs', 'single_choice', 124 * 6],
        "chinese_modern_lit": ['2010-2022_Chinese_Modern_Lit', 'multi_question_choice', 87 * 3],
        "chinese_lang_and_usage": ['2010-2022_Chinese_Lang_and_Usage_MCQs', 'multi_question_choice', 80 * 3],
        "english": ['2010-2013_English_MCQs', 'single_choice', 105 * 1],
        "english_cloze_test": ['2012-2022_English_Cloze_Test', 'five_out_of_seven', 26 * 2],
        "english_fill_in_blanks": ['2010-2022_English_Fill_in_Blanks', 'multi_question_choice', 600 * 1.5],
        "english_reading_comp": ['2010-2022_English_Reading_Comp', 'multi_question_choice', 470 * 2],
        "geography": ['2010-2022_Geography_MCQs', 'multi_question_choice', 95 * 4],
        "history": ['2010-2022_History_MCQs', 'single_choice', 287 * 4],
        "physics": ['2010-2022_Physics_MCQs', 'multi_choice', 64 * 6],
        "politics": ['2010-2022_Political_Science_MCQs', 'single_choice', 320 * 4],
}


TASK_MAPPING = {
    "bbh": ["boolean_expressions",
            "causal_judgement",
            "date_understanding",
            "disambiguation_qa",
            "dyck_languages",
            "formal_fallacies",
            "geometric_shapes",
            "hyperbaton",
            "logical_deduction_five_objects",
            "logical_deduction_seven_objects",
            "logical_deduction_three_objects",
            "movie_recommendation",
            "multistep_arithmetic_two",
            "navigate",
            "object_counting",
            "penguins_in_a_table",
            "reasoning_about_colored_objects",
            "ruin_names",
            "salient_translation_error_detection",
            "snarks",
            "sports_understanding",
            "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects",
            "tracking_shuffled_objects_three_objects",
            "web_of_lies",
            "word_sorting"],
    "agieval": [
            "gaokao_chinese",
            "gaokao_geography",
            "gaokao_history",
            "gaokao_biology",
            "gaokao_chemistry",
            "gaokao_physics",
            "gaokao_mathqa",
            "gaokao_english",
            "gaokao_mathcloze",
            "sat_math",
            "sat_en", "aqua_rat",
            "lsat_ar", "lsat_lr", "lsat_rc",
            "logiqa_en", "logiqa_zh",
            #"jec_qa_kd", "jec_qa_ca",
            "math",
            "sat_en_without_passage"],
    "ceval": list(CEVAL_CATS.keys()),
    "mmlu": list(MMLU_CATS.keys()),
    "cmmlu": list(CMMLU_CATS.keys()),
    "gaokao": list(GAOKAO_CATS.keys()),
}


def bbh_post_metrics(task, all_metrics):
    results = []
    for k, v in all_metrics.items():
        if task in k:
            results.append(v["accuracy"])
    metrics = {"average": np.mean(results)}
    return metrics


def ceval_post_metrics(task, all_metrics):
    summary_cats = ["stem", "social science", "humanities", "other", "hard", "average"]
    cat_scores = dict(zip(summary_cats, np.zeros((len(summary_cats), 2))))

    for k, v in all_metrics.items():
        if task not in k:
            continue
        subs = CEVAL_CATS[k.split("/")[1]]
        lens = subs[3]
        all_scores = v["accuracy"] * lens

        cat_scores[subs[2].lower()][0] += all_scores
        cat_scores[subs[2].lower()][1] += lens
        cat_scores["average"][0] += all_scores
        cat_scores["average"][1] += lens

        if "math" in k or "physics" in k or "chemistry" in k:
            cat_scores["hard"][0] += all_scores
            cat_scores["hard"][1] += lens

    metrics = {k: v[0] / v[1] for k, v in cat_scores.items()}
    return metrics


def mmlu_post_metrics(task, all_metrics):
    summary_cats = [i[0] for i in MMLU_CATS.values()]
    summary_cats.extend(["STEM", "social sciences",
        "humanities", "other (business, health, misc.)", "average"])
    cat_scores = dict(zip(summary_cats, np.zeros((len(summary_cats), 2))))

    for k, v in all_metrics.items():
        if task not in k:
            continue
        subs = MMLU_CATS[k.split("/")[1]]
        lens = subs[-1]
        all_scores = v["accuracy"] * lens

        cat_scores[subs[0]][0] += all_scores
        cat_scores[subs[0]][1] += lens
        cat_scores[subs[1]][0] += all_scores
        cat_scores[subs[1]][1] += lens
        cat_scores["average"][0] += all_scores
        cat_scores["average"][1] += lens

    metrics = {k: v[0] / v[1] for k, v in cat_scores.items()}
    return metrics


def cmmlu_post_metrics(task, all_metrics):
    summary_cats = [i[1] for i in CMMLU_CATS.values()]
    summary_cats.extend(["STEM", "Social Science",
        "Humanities", "Other", "China specific", "average"])
    cat_scores = dict(zip(summary_cats, np.zeros((len(summary_cats), 2))))

    for k, v in all_metrics.items():
        if task not in k:
            continue
        subs = CMMLU_CATS[k.split("/")[1]]
        lens = subs[-1]
        all_scores = v["accuracy"] * lens

        cat_scores[subs[1]][0] += all_scores
        cat_scores[subs[1]][1] += lens
        cat_scores[subs[2]][0] += all_scores
        cat_scores[subs[2]][1] += lens
        if k.split("/")[1] in CMMLU_CHINESE_SPECIFIC:
            cat_scores["China specific"][0] += all_scores
            cat_scores["China specific"][1] += lens
        cat_scores["average"][0] += all_scores
        cat_scores["average"][1] += lens

    metrics = {k: v[0] / v[1] for k, v in cat_scores.items()}
    return metrics


def agieval_post_metrics(task, all_metrics):
    cat_scores = {"gaokao_average": [], "average": []}
    for k, v in all_metrics.items():
        if task in k:
            cat_scores["average"].append(v["accuracy"])
            if "gaokao" in k:
                cat_scores["gaokao_average"].append(v["accuracy"])
    metrics = {k: np.mean(v) for k, v in cat_scores.items()}
    return metrics


def gaokao_post_metrics(task, all_metrics):
    name_mapping = {
            "A": ["english", "math1", "chinese", "physics", "chemistry", "biology"],
            "B": ["english", "math2", "chinese", "history", "geography", "politics"],
            "score": ["english", "math1", "math2", "chinese", "physics",
                    "chemistry", "biology", "history", "geography", "politics"]
    }
    scores_mapping = {
            "A": [150, 150, 150, 110, 100, 90],
            "B": [150, 150, 150, 100, 100, 100],
            "score": [150, 150, 150, 150, 100, 100, 100, 100, 100, 100]
    }
    cat_scores = {"A": np.zeros((6, 2)), "B": np.zeros((6, 2)), "score": np.zeros((10, 2))}
    for k, v in all_metrics.items():
        if not k.startswith(task):
            continue
        cat_name = k.split("/")[-1]
        scores = GAOKAO_CATS[cat_name][-1]
        for i in cat_scores.keys():
            try:
                cur_idx = name_mapping[i].index(cat_name.split("_")[0])
                cat_scores[i][cur_idx][0] += v["acurracy"] * scores
                cat_scores[i][cur_idx][1] += scores
            except:
                pass

    metrics = {}
    for i in cat_scores.keys():
        metrics[i] = np.sum(cat_scores[i][:, 0]/cat_scores[i][:, 1]*np.array(scores_mapping[i])) * 0.01
    return metrics


METRICS_MAPPING = {
        "bbh": bbh_post_metrics,
        "ceval": ceval_post_metrics,
        "mmlu": mmlu_post_metrics,
        "cmmlu": cmmlu_post_metrics,
        "agieval": agieval_post_metrics,
        "gaokao": gaokao_post_metrics,
}

BASE_TASKS = [
        "boolq", "piqa", "siqa", "sciq", "hellaswag",
        "winogrande", "arc_e", "arc_c", "obqa", "race_m",
        "race_h", "triviaqa", "naturalqa", "drop_gen",
        "clue_c3", "clue_wsc", "clue_cmrc", "xtreme",
]
for task in TASK_MAPPING.keys():
    BASE_TASKS.extend(["/".join([task, i]) for i in TASK_MAPPING[task]])


def dataset_mapping(tasks):
    for k in TASK_MAPPING.keys():
        if k in tasks:
            tasks.remove(k)
            tasks.extend(["/".join([k, i]) for i in TASK_MAPPING[k]])
    return tasks


def metrics_post_process(tasks, all_metrics):
    metrics_to_write = {}
    for k in TASK_MAPPING.keys():
        if k in tasks:
            metrics = METRICS_MAPPING[k](k, all_metrics)
            scores_log = ", ".join("{} {:.2%}".format(m, c) for m, c in metrics.items())
            logger.info(f'Summary {scores_log} - {k}')
            metrics_to_write[k] = metrics
    return metrics_to_write
