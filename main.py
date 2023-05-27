import openai

from utils import (
    apply_heuristics,
    heuristic_adjust_spaces,
    get_bleu_and_codebleu,
    get_env_variable,
    get_predictions_from_openai_and_write_to_file,
    read_raw_tufano_dataset_from_csv,
    write_list_to_file,
    read_dataset
)

if __name__ == "__main__":
    openai.api_key = get_env_variable("OPENAI_API_KEY")

    OUTPUT_DIRECTORY = "outputs"
    TUFANO_SOURCE_FILE_PATH = "datasets/tufano/test_CC_src.txt"
    TUFANO_TARGET_FILE_PATH = "datasets/tufano/test_CC_tgt.txt"
    TUFANO_RAW_DATASET_FILE_PATH = "datasets/tufano/raw_test.csv"
    R4R_SOURCE_FILE_PATH = "datasets/R4R/test_CC_src.txt"
    R4R_TARGET_FILE_PATH = "datasets/R4R/test_CC_tgt.txt"

    START_INDEX = 0
    END_INDEX = 100
    code_reviews, buggy_codes, target_codes = read_dataset(
        dataset_name="R4R", source_file_path=R4R_SOURCE_FILE_PATH, target_file_path=R4R_TARGET_FILE_PATH
    )

    # code_reviews, buggy_codes, target_codes = read_raw_tufano_dataset_from_csv(TUFANO_RAW_DATASET_FILE_PATH)

    # get_predictions_from_openai_and_write_to_file(
    #     f"{OUTPUT_DIRECTORY}/r4r_predictions_raw_no_heuristic.txt",
    #     f"{OUTPUT_DIRECTORY}/r4r_ground_truths_raw_no_heuristic.txt",
    #     code_reviews,
    #     buggy_codes,
    #     target_codes,
    #     START_INDEX,
    #     END_INDEX
    # )
    
    print("************** WITHOUT HEURISTICS RESULT *******************")
    get_bleu_and_codebleu(
        "outputs/r4r_ground_truths_raw_no_heuristic_0_99_formatted.txt",
        "outputs/r4r_predictions_raw_no_heuristic_0_99_formatted.txt",
    )

    # print("************** WITHOUT HEURISTICS RESULT *******************")
    # get_bleu_and_codebleu(
    #     "outputs/tufano_ground_truths_raw_no_heuristic_0_1718.txt",
    #     "outputs/tufano_predictions_raw_no_heuristics_formatted.txt",
    # )

    # print("************** WITH HEURISTICS RESULT *******************")
    # get_bleu_and_codebleu("outputs/tufano_ground_truths_raw.txt", "outputs/tufano_predictions_raw.txt")

    # transfer_content_to_another_file(
    #     keyword="response:",
    #     input_file="logs/LOGS_tufano_predictions_raw_1500_1718.txt",
    #     output_file="outputs/tufano_predictions_raw_1500_1718.txt",
    # )
    # transfer_content_to_another_file(
    #     keyword="target code:",
    #     input_file="logs/LOGS_tufano_predictions_raw_1500_1718.txt",
    #     output_file="outputs/tufano_ground_truths_raw_1500_1718.txt",
    # )

    # combine_output_files("ground_truths", "outputs", "tufano_ground_truths_raw.txt")
    # combine_output_files("predictions", "outputs", "tufano_predictions_raw.txt")

    # with open("outputs/r4r_predictions_raw_no_heuristic_0_99.txt", "r", encoding="UTF-8") as input_file:
    #     input_lines = input_file.readlines()
    #     output_lines = []
    #     for line in input_lines:
    #         output_line = apply_heuristics(line)
    #         output_lines.append(output_line)
    #     write_list_to_file("outputs/r4r_predictions_raw_no_heuristic_0_99_formatted.txt", output_lines)
