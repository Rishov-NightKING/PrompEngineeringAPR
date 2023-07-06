import os
import re
import subprocess
import time

import openai
import pandas as pd

from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def prompt_response(system_prompt, user_prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": f"{system_prompt}"}, {"role": "user", "content": f"{user_prompt}"}],
        temperature=0,
        max_tokens=500,  # for tufano - 200, R4R - 500
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    response_message = response["choices"][0]["message"]["content"]
    response_message = response_message.replace("\n", " ")

    return response_message


def remove_extra_spaces(line):
    while True:
        line = line.replace("  ", " ")
        if "  " not in line:
            break
    return line.strip()


def adjust_spaces(text):
    # Create a set of all the items to check for membership
    first_occurrence_list = [
        "%",
        "&",
        "?",
        "<",
        ">",
        ",",
        ":",
        ";",
        ".",
        "!",
        "^",
        "+",
        "-",
        "/",
        "*",
        "=",
    ]
    second_occurrence_list = ["=", "+", "-", "/", "*", "&"]
    brackets = ["(", ")", "{", "}", "[", "]"]
    # Initialize the output string
    output = ""
    # Initialize the current index to 0
    i = 0
    # Loop over the characters in the text
    while i < len(text):
        # Check if the current character is in the item set
        if text[i] in first_occurrence_list:
            # If it is, add it to the output string with a space on either side

            # If the current character is a '<', check if the next character is also in the item set
            if i + 1 < len(text) and text[i + 1] in second_occurrence_list:
                # If it is, add it to the output string as a single unit with a space on either side
                output += " " + text[i] + text[i + 1] + " "
                # Move the current index forward by two characters
                i += 2
            else:
                output += " " + text[i] + " "
                # Otherwise, move the current index forward by one character
                i += 1
        elif text[i] in brackets:
            output += " " + text[i] + " "
            i += 1
        else:
            # If the current character is not in the item set, add it to the output string
            output += text[i]
            # Move the current index forward by one character
            i += 1
    output = remove_extra_spaces(output)
    # Return the output string
    return output


def heuristic_remove_redundant_words(line):
    redundant_words = [
        "Here's the",
        "Code Refactored :",
        "Refactored code :",
        "// Refactored code without comments",
        "// Refactored code",
        "Updated code :",
        "Fixed code :",
        "Corrected code :",
        "Code :",
        "```",
        "< START >",
        "< END >",
        "Code after refactoring :" "refactored code based on the review :",
    ]
    for redundant_word in redundant_words:
        line = line.replace(redundant_word, "").replace(redundant_word.lower(), "").replace(redundant_word.title(), "")
    return line.strip()


def heuristic_remove_starts_with_java(line):
    if line.startswith("java"):
        line = line[4:]
    return line.strip()


def heuristic_remove_code_explanation_at_the_end(line):
    end_words = [
        "Explanation :",
        "Note :",
        "Reasoning :",
        "In the refactored code",
        "The refactored code",
        "Changes made :",
        "Changes Made :",
        "Refactored Review :",
    ]
    for end_word in end_words:
        line = line.split(end_word)[0]

    return line.strip()


def apply_heuristics(line):
    heuristics = [
        adjust_spaces,
        heuristic_remove_redundant_words,
        heuristic_remove_starts_with_java,
        heuristic_remove_code_explanation_at_the_end,
        remove_extra_spaces,
    ]

    for heuristic in heuristics:
        line = heuristic(line)

    return line


def apply_heuristic_in_file(input_file):
    with open(input_file, "r", encoding="UTF-8") as f1:
        lines = f1.readlines()

    lines = [apply_heuristics(line) for line in lines]
    output_file = f"{input_file.split('.')[0]}_applied_heuristics.txt"
    write_list_to_file(output_file, lines)


def modify_file_name(file_name, start_index, end_index):
    file_name_parts = file_name.split(".")
    file_name = f"{file_name_parts[0]}_{start_index}_{end_index - 1}.{file_name_parts[1]}"
    return file_name


def write_list_to_file(file_name, list_name, start_index=0, end_index=None):
    if end_index is None:
        end_index = len(list_name)
    file = open(file_name, "w", encoding="UTF-8")
    file.writelines([item + "\n" for item in list_name[start_index:end_index]])
    file.close()


def read_env_file(file_path):
    env_variables = {}
    with open(file_path, "r", encoding="UTF-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=")
                env_variables[key] = value
    return env_variables


def get_env_variable(key, file_path=".env"):
    env_variables = read_env_file(file_path)
    return env_variables.get(key)


def modify_R4R_dataset(buggy_code, target):
    start_focus_tag = "<|startfocus|>"
    end_focus_tag = "<|endfocus|>"
    first_end_point = buggy_code.index(start_focus_tag)
    second_end_point = buggy_code.index(end_focus_tag) + len(end_focus_tag)

    before_context = buggy_code[:first_end_point]
    after_context = buggy_code[second_end_point:]

    # handling del targets
    if target.strip() == "<|del|>":
        target = ""

    output = before_context + target.strip() + after_context

    # handling extra spaces for del targets
    output = remove_extra_spaces(output)

    return output


def modify_R4R_for_EM(buggy_code, target):
    start_focus_tag = "<|startfocus|>"
    end_focus_tag = "<|endfocus|>"
    first_end_point = buggy_code.index(start_focus_tag) + len(start_focus_tag)
    second_end_point = buggy_code.index(end_focus_tag)
    focus_part = buggy_code[first_end_point:second_end_point].strip()
    target = target.strip()
    if target.startswith("<|del|>"):
        target += focus_part
    return target


def heuristic_count_frequency(target_substring, target_string):
    return target_string.count(target_substring)


def get_EM(pred_file, ref_file, dataset):
    with open(ref_file, "r", encoding="UTF-8") as f1, open(pred_file, "r", encoding="UTF-8") as f2:
        refs = f1.readlines()
        preds = f2.readlines()

    matches = []
    matches_r_equal_p = []
    for i, (r, p) in enumerate(zip(refs, preds)):
        r, p = r.strip(), p.strip()
        if r in p:
            matches.append(i)
        if r == p:
            matches_r_equal_p.append(i)

    if dataset == "R4R":
        print(f"EM: {len(matches) / len(refs) * 100:.2f}%")
    elif dataset == "tufano" :
        print(f"EM: {len(matches_r_equal_p) / len(refs) * 100:.2f}%")


def get_EM_R4R(pred_file, ref_file):
    with open(ref_file, "r", encoding="UTF-8") as f1, open(pred_file, "r", encoding="UTF-8") as f2:
        refs = f1.readlines()
        preds = f2.readlines()

    count = 0
    matches = []
    del_matches = []
    possible_duplicates = []
    matches_r_equal_p = []
    del_token = "< |del| >"
    focus_null = 0
    for i, (r, p) in enumerate(zip(refs, preds)):
        r, p = r.strip(), p.strip()
        if r.startswith(del_token):
            focus_part = r[len(del_token):]
            if focus_part == "":
                count += 1
                focus_null += 1
            elif heuristic_count_frequency(focus_part, p) == 0:
                count += 0
                del_matches.append(i)
        if r in p:
            if heuristic_count_frequency(r, p) > 1:
                possible_duplicates.append(i)
            count += 1
            matches.append(i)
        if r == p:
            matches_r_equal_p.append(i)

    print(f"EM: {len(matches) / len(refs) * 100:.2f}%")
    print(f"possible del matches: {del_matches}, {len(del_matches)}, {len(matches)}")


def read_dataset(dataset_name, source_file_path, target_file_path):
    with open(source_file_path, "r", encoding="UTF-8") as src_file, open(
            target_file_path, "r", encoding="UTF-8"
    ) as tgt_file:
        source_codes = src_file.readlines()
        target_codes = tgt_file.readlines()

    buggy_codes = []
    code_reviews = []
    modified_target_codes = []
    # targets_modified_for_EM = []
    for code, target_code in zip(source_codes, target_codes):
        start_comment_tag = "<|startcomment|>"
        end_comment_tag = "<|endcomment|>"
        end_point = code.index(end_comment_tag) + len(end_comment_tag)

        code_review = code[:end_point].replace(start_comment_tag, "").replace(end_comment_tag, "")
        buggy_code = code[end_point + 1:].replace("\n", "")

        code_reviews.append(code_review)
        buggy_codes.append(buggy_code)
        if dataset_name == "R4R":
            full_target_code = modify_R4R_dataset(buggy_code, target_code)
            # target_modified_for_EM = modify_R4R_for_EM(buggy_code, target_code)
            modified_target_codes.append(full_target_code)
            # targets_modified_for_EM.append(target_modified_for_EM)

    if dataset_name == "tufano":
        return code_reviews, buggy_codes, target_codes
    elif dataset_name == "R4R":
        # write_list_to_file("outputs/r4r_ground_truth_paths_modified_for_EM.txt", targets_modified_for_EM)
        return code_reviews, buggy_codes, modified_target_codes


def read_raw_tufano_dataset_from_csv(file_path):
    df = pd.read_csv(file_path)
    code_reviews = list(df["comment"])
    code_reviews = [
        code_review.replace("\n", " ").replace("\t", " ").replace("\r", " ") for code_review in code_reviews
    ]
    code_reviews = [remove_extra_spaces(code_review) for code_review in code_reviews]

    buggy_codes = list(df["before_marked"])
    buggy_codes = [
        buggy_code.replace("START", "<START>")
        .replace("END", "<END>")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\r", " ")
        for buggy_code in buggy_codes
    ]
    buggy_codes = [remove_extra_spaces(buggy_code) for buggy_code in buggy_codes]

    target_codes = list(df["after"])
    target_codes = [
        target_code.replace("\n", " ").replace("\t", " ").replace("\r", " ") for target_code in target_codes
    ]
    target_codes = [adjust_spaces(target_code) for target_code in target_codes]

    # write raw tufano csv file data to text files
    # raw_test_cc_src = [
    #     f"<|startcomment|> {code_review} <|endcomment|> {buggy_code}"
    #     for code_review, buggy_code in zip(code_reviews, buggy_codes)
    # ]
    # write_list_to_file("datasets/tufano/raw_test_CC_src.txt", raw_test_cc_src)
    # write_list_to_file("datasets/tufano/raw_test_CC_tgt.txt", target_codes)

    return code_reviews, buggy_codes, target_codes


def run_python_file(bleu_type, python_file_path, predictions_file_path, ground_truths_file_path, lang="java"):
    # Arguments to pass to the Python file
    arguments = []
    if bleu_type == "BLEU":
        arguments = ["--references", ground_truths_file_path, "--predictions", predictions_file_path]
    elif bleu_type == "CodeBLEU":
        arguments = ["--refs", ground_truths_file_path, "--hyp", predictions_file_path, "--lang", lang]

    try:
        # Run the Python file with arguments
        # if bleu_type == "CodeBLEU":
        #     tree_sitter_build_bash_path = "evaluation/CodeBLEU/parser/build.sh"
        #     subprocess.run(["bash", tree_sitter_build_bash_path], check=True)
        subprocess.run(["python", python_file_path] + arguments, check=True)
        # print("Python file executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while executing Python file {python_file_path}: {e}")


def get_bleu_and_codebleu(prediction_file_path, ground_truth_path):
    # calculate BLEU
    run_python_file(
        "BLEU",
        "evaluation/bleu.py",
        prediction_file_path,
        ground_truth_path,
    )
    # calculate CodeBLEU
    run_python_file(
        "CodeBLEU",
        "evaluation/CodeBLEU/calc_code_bleu.py",
        prediction_file_path,
        ground_truth_path,
    )


def get_predictions_from_openai_and_write_to_file(
        prediction_file_path, ground_truth_path, code_reviews, buggy_codes, target_codes, start_index=0, end_index=None
):
    if end_index is None:
        end_index = len(target_codes)

    system_prompt = "You are a coding assistant. You generate only the source code."
    user_command = "Refactor the Buggy Code using the Review without comments"

    prediction_list = []

    log_file_name = (
        f"logs/LOGS_{prediction_file_path.split('/')[1].replace('.txt', '')}_{start_index}_{end_index - 1}.txt"
    )
    log_file = open(log_file_name, "w", encoding="UTF-8")

    i = start_index
    while i < end_index:
        try:
            buggy_code = buggy_codes[i]
            code_review = code_reviews[i]
            target_code = target_codes[i]

            user_prompt = f"Buggy Code: {buggy_code}\nReview: {code_review}\n{user_command}"
            prediction = prompt_response(system_prompt, user_prompt)

            # apply all heuristics
            # prediction = apply_heuristics(prediction)
            prediction = adjust_spaces(prediction)
            prediction = remove_extra_spaces(prediction)
            prediction_list.append(prediction)

            SAMPLE_NO = f"sample: {i}"
            BUGGY_CODE = f"buggy_code: {buggy_code}"
            CODE_REVIEW = f"code_review: {code_review}"
            TARGET_CODE = f"target code: {target_code}"
            PREDICTION = f"response: {prediction}"

            print(SAMPLE_NO)
            print(BUGGY_CODE)
            print(CODE_REVIEW)
            print(TARGET_CODE)
            print(PREDICTION)
            print()

            log_file.write(SAMPLE_NO + "\n")
            log_file.write(BUGGY_CODE + "\n")
            log_file.write(CODE_REVIEW + "\n")
            log_file.write(TARGET_CODE + "\n")
            log_file.write(PREDICTION + "\n")
            log_file.write("\n")

            time.sleep(20)
            i += 1
        except Exception as e:
            print(f"An Exception occurred at sample: {i}. Error details: {str(e)}")
            time.sleep(60)

    prediction_file_path = modify_file_name(prediction_file_path, start_index, end_index)
    ground_truth_path = modify_file_name(ground_truth_path, start_index, end_index)
    # write predictions to a file
    write_list_to_file(file_name=prediction_file_path, list_name=prediction_list)
    # write ground truths to a file
    write_list_to_file(
        file_name=ground_truth_path, list_name=target_codes, start_index=start_index, end_index=end_index
    )
    # calculate BLEU and CodeBLEU
    get_bleu_and_codebleu(prediction_file_path, ground_truth_path)


def transfer_content_to_another_file(keyword, input_file, output_file):
    input = open(input_file, "r", encoding="UTF-8")
    input_lines = input.readlines()

    output_lines = []
    for input_line in input_lines:
        if input_line.startswith(keyword):
            output_line = input_line.split(keyword)[1].strip()
            output_lines.append(output_line)

    write_list_to_file(file_name=output_file, list_name=output_lines)


def tryint(s):
    """
    Return an int if possible, or `s` unchanged.
    """
    try:
        return int(s)
    except ValueError:
        return s


def combine_output_files(keyword, directory_path, combined_file_name):
    file_list = [f for f in os.listdir(directory_path) if keyword in f]
    file_list = sorted(file_list, key=lambda file_name: [tryint(c) for c in re.split("([0-9]+)", file_name)])

    # for file in file_list:
    #     print(os.path.join(directory_path, file))

    combined_file = open(os.path.join(directory_path, combined_file_name), "w", encoding="UTF-8")
    for file in file_list:
        input_file = open(os.path.join(directory_path, file), "r", encoding="UTF-8")
        input_lines = input_file.readlines()
        combined_file.writelines(input_lines)

    combined_file.close()


def format_file(filename, function_name):
    with open(filename, "r", encoding="UTF-8") as file:
        input = file.readlines()
        output_lines = []

        for line in input:
            output_line = function_name(line)
            output_lines.append(output_line)

        output_file_name = f"{filename.split('.')[0]}_formatted.txt"
        write_list_to_file(f"{output_file_name}", output_lines)


def get_buggy_code_contexts(buggy_codes):
    buggy_code_contexts = []
    for buggy_code in buggy_codes:
        index = buggy_code.index("<START>")
        buggy_code_context = buggy_code[:index]
        buggy_code_contexts.append(buggy_code_context)
    return buggy_code_contexts


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_hugging_face_model(hugging_face_model):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)
    model = AutoModelForCausalLM.from_pretrained(hugging_face_model)
    model.to(device)
    return model, tokenizer


def get_predictions_from_generative_model(model, tokenizer, code_reviews, buggy_code_contexts):
    device = get_device()
    for i, (code_review, buggy_code_context) in enumerate(zip(code_reviews, buggy_code_contexts)):
        user_prompt = f"{code_review}\n{buggy_code_context}"
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(device)
        result = model.generate(
            input_ids,
            max_length=200,
            num_beams=5,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0001,
            early_stopping=True,
        )
        output = tokenizer.decode(result, skip_special_tokens=True)
        print(f"Sample: {i}")
        print(output)
        print("End prediction*****************")


def prompt_response_edit_api(code_review, buggy_code):
    response = openai.Edit.create(
        model="code-davinci-edit-001",
        input=f"{buggy_code}",
        instruction=f"Refactor the code using the Review: {code_review}",
        temperature=0,
        top_p=1,
    )
    response_message = response["choices"][0]["text"]
    response_message = response_message.replace("\n", " ")

    return response_message.strip()


def get_predictions_from_edit_api_and_write_to_file(
        prediction_file_path, ground_truth_path, code_reviews, buggy_codes, target_codes, start_index=0, end_index=None
):
    if end_index is None:
        end_index = len(target_codes)

    prediction_list = []

    log_file_name = (
        f"logs/EDIT_LOGS_{prediction_file_path.split('/')[1].replace('.txt', '')}_{start_index}_{end_index}.txt"
    )
    log_file = open(log_file_name, "w", encoding="UTF-8")

    i = start_index
    error_count = 0
    while i <= end_index:
        try:
            buggy_code = buggy_codes[i]
            code_review = code_reviews[i]
            target_code = target_codes[i]

            prediction = prompt_response_edit_api(code_review, buggy_code)

            # apply all heuristics
            # prediction = apply_heuristics(prediction)
            prediction = remove_extra_spaces(prediction)
            prediction_list.append(prediction)

            SAMPLE_NO = f"sample: {i}"
            BUGGY_CODE = f"buggy_code: {buggy_code}"
            CODE_REVIEW = f"code_review: {code_review}"
            TARGET_CODE = f"target code: {target_code}"
            PREDICTION = f"response: {prediction}"

            print(SAMPLE_NO)
            print(BUGGY_CODE)
            print(CODE_REVIEW)
            print(TARGET_CODE)
            print(PREDICTION)
            print()

            log_file.write(SAMPLE_NO + "\n")
            log_file.write(BUGGY_CODE + "\n")
            log_file.write(CODE_REVIEW + "\n")
            log_file.write(TARGET_CODE + "\n")
            log_file.write(PREDICTION + "\n")
            log_file.write("\n")
            i += 1
        except Exception as e:
            error_count += 1
            print(f"An Exception occurred at sample: {i}. Error details: {str(e)}")
            time.sleep(10)
            if error_count == 5:
                i += 1
                error_count = 0

    prediction_file_path = modify_file_name(prediction_file_path, start_index, end_index)
    ground_truth_path = modify_file_name(ground_truth_path, start_index, end_index)
    # write predictions to a file
    write_list_to_file(file_name=prediction_file_path, list_name=prediction_list)
    # write ground truths to a file
    write_list_to_file(
        file_name=ground_truth_path, list_name=target_codes, start_index=start_index, end_index=end_index
    )
    # calculate BLEU and CodeBLEU
    get_bleu_and_codebleu(prediction_file_path, ground_truth_path)


def vectorize_and_dump_to_file(train_data, output_file_name):
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_data)

    save_npz(output_file_name, train_vectors)


def load_vectorized_data(file_name):
    loaded_vectors = load_npz(file_name)
    return loaded_vectors


def get_few_shot_predictions_from_openai_and_write_to_file(
        prediction_file_path: str,
        ground_truth_path: str,
        train_dataset: tuple,
        test_dataset: tuple,
        top_k: int,
        start_index: int = 0,
        end_index: int = None,
):
    train_code_reviews, train_buggy_codes, train_target_codes = train_dataset
    test_code_reviews, test_buggy_codes, test_target_codes = test_dataset

    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_code_reviews)
    test_vectors = vectorizer.transform(test_code_reviews)
    # calculate cosine similarity for all the test samples with train samples
    similarities = cosine_similarity(test_vectors, train_vectors)

    if end_index is None:
        end_index = len(test_target_codes)

    system_prompt = "You are a coding assistant. You generate only the source code."
    user_command = "Refactor the Buggy Code using the Review without comments"

    prediction_list = []

    log_file_name = (
        f"logs/FEW_SHOT_LOGS_{prediction_file_path.split('/')[1].replace('.txt', '')}_{start_index}_{end_index - 1}.txt"
    )
    log_file = open(log_file_name, "w", encoding="UTF-8")

    i = start_index
    while i < end_index:
        try:
            # few shot user prompt creation
            top_k_samples = [
                index for index, score in sorted(enumerate(similarities[i]), key=lambda x: x[1], reverse=True)[:top_k]
            ]
            user_prompt = ""
            for sample_index in top_k_samples:
                train_buggy_code = train_buggy_codes[sample_index]
                train_code_review = train_code_reviews[sample_index]
                train_target_code = train_target_codes[sample_index]
                user_prompt += (
                    f"Buggy Code: {train_buggy_code}\nReview: {train_code_review}\nFixed Code: {train_target_code}\n\n"
                )

            test_buggy_code = test_buggy_codes[i]
            test_code_review = test_code_reviews[i]
            test_target_code = test_target_codes[i]
            user_prompt += f"Buggy Code: {test_buggy_code}\nReview: {test_code_review}\n{user_command}"
            prediction = prompt_response(system_prompt, user_prompt)

            # apply all heuristics
            # prediction = apply_heuristics(prediction)
            prediction = remove_extra_spaces(prediction)
            # prediction = adjust_spaces(prediction)
            prediction_list.append(prediction)

            SAMPLE_NO = f"sample: {i}"
            BUGGY_CODE = f"buggy_code: {test_buggy_code}"
            CODE_REVIEW = f"code_review: {test_code_review}"
            TARGET_CODE = f"target code: {test_target_code}"
            PREDICTION = f"response: {prediction}"

            print(SAMPLE_NO)
            print(f"user prompt: \n{user_prompt}\n\n")
            print(BUGGY_CODE)
            print(CODE_REVIEW)
            print(TARGET_CODE)
            print(PREDICTION)
            print()

            log_file.write(SAMPLE_NO + "\n")
            log_file.write(BUGGY_CODE + "\n")
            log_file.write(CODE_REVIEW + "\n")
            log_file.write(TARGET_CODE + "\n")
            log_file.write(PREDICTION + "\n")
            log_file.write("\n")

            time.sleep(17)
            i += 1
        except Exception as e:
            print(f"An Exception occurred at sample: {i}. Error details: {str(e)}")
            time.sleep(30)

    prediction_file_path = modify_file_name(prediction_file_path, start_index, end_index)
    ground_truth_path = modify_file_name(ground_truth_path, start_index, end_index)
    # write predictions to a file
    write_list_to_file(file_name=prediction_file_path, list_name=prediction_list)
    # write ground truths to a file
    write_list_to_file(
        file_name=ground_truth_path, list_name=test_target_codes, start_index=start_index, end_index=end_index
    )
    # calculate BLEU and CodeBLEU
    get_bleu_and_codebleu(prediction_file_path, ground_truth_path)
