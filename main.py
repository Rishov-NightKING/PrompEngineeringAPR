import openai

from utils import (
    apply_heuristics,
    apply_heuristic_in_file,
    heuristic_adjust_spaces,
    get_bleu_and_codebleu,
    get_env_variable,
    get_predictions_from_openai_and_write_to_file,
    read_raw_tufano_dataset_from_csv,
    write_list_to_file,
    read_dataset,
    get_EM_R4R,
    format_file,
    transfer_content_to_another_file,
    get_predictions_from_edit_api_and_write_to_file
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
    END_INDEX = None
    # code_reviews, buggy_codes, target_codes = read_dataset(
    #     dataset_name="R4R", source_file_path=R4R_SOURCE_FILE_PATH, target_file_path=R4R_TARGET_FILE_PATH
    # )

    # code_reviews, buggy_codes, target_codes = read_raw_tufano_dataset_from_csv(TUFANO_RAW_DATASET_FILE_PATH)

    # get_predictions_from_edit_api_and_write_to_file(
    #     f"{OUTPUT_DIRECTORY}/edit_r4r_predictions_raw_no_heuristic.txt",
    #     f"{OUTPUT_DIRECTORY}/edit_r4r_ground_truths_raw_no_heuristic.txt",
    #     code_reviews,
    #     buggy_codes,
    #     target_codes,
    #     start_index=START_INDEX,
    #     end_index=END_INDEX
    # )

    # get_predictions_from_openai_and_write_to_file(
    #     f"{OUTPUT_DIRECTORY}/r4r_predictions_raw_no_heuristic.txt",
    #     f"{OUTPUT_DIRECTORY}/r4r_ground_truths_raw_no_heuristic.txt",
    #     code_reviews,
    #     buggy_codes,
    #     target_codes
    # )

    # print("************** WITHOUT HEURISTICS RESULT *******************")
    # get_bleu_and_codebleu(
    #     "outputs/r4r_ground_truths_raw_no_heuristic_0_2954_formatted.txt",
    #     "outputs/r4r_predictions_raw_no_heuristic_0_2954_formatted.txt",
    # )

    # print("************** BEFORE WITHOUT HEURISTICS RESULT *******************")
    # get_bleu_and_codebleu(
    #     "outputs/tufano_ground_truths_raw_no_heuristic_0_1718.txt",
    #     "outputs/tufano_predictions_raw_no_heuristics_formatted.txt",
    # )
    #
    # print("************** BEFORE WITH HEURISTICS RESULT *******************")
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

    # format_file("outputs/r4r_ground_truths_raw_no_heuristic_0_2954.txt", heuristic_adjust_spaces)
    # format_file("outputs/r4r_predictions_raw_no_heuristic_0_2954.txt", apply_heuristics)

    # format_file("outputs/r4r_ground_truth_paths_modified_for_EM.txt", heuristic_adjust_spaces)
    # get_EM_R4R(
    #     "outputs/r4r_ground_truths_raw_no_heuristic_0_2954_formatted.txt",
    #     "outputs/r4r_predictions_raw_no_heuristic_0_2954_formatted.txt",
    # )
    # get_EM_R4R(
    #     "outputs/r4r_ground_truth_paths_modified_for_EM_formatted.txt",
    #     "outputs/r4r_predictions_raw_no_heuristic_0_2954_formatted.txt",
    # )

    # transfer_content_to_another_file(
    #     keyword="response:",
    #     input_file="logs/uninterrupted_LOGS_tufano_predictions_raw_no_heuristic_0_1718.txt",
    #     output_file="outputs/tufano_predictions_raw_from_log.txt",
    # )
    # transfer_content_to_another_file(
    #     keyword="target code:",
    #     input_file="logs/uninterrupted_LOGS_tufano_predictions_raw_no_heuristic_0_1718.txt",
    #     output_file="outputs/tufano_ground_truths_raw_from_log.txt",
    # )

    # format_file("outputs/tufano_predictions_raw_from_log.txt", apply_heuristics)

    # get_EM_R4R(
    #     "outputs/tufano_ground_truths_raw_from_log.txt",
    #     "outputs/tufano_predictions_raw_from_log_no_heuristics.txt",
    # )
    # #
    print("************** AFTER WITHOUT HEURISTICS RESULT *******************")
    get_bleu_and_codebleu(
        "outputs/tufano_ground_truths_raw_from_log.txt",
        "outputs/tufano_predictions_raw_from_log_no_heuristics.txt",
    )
    # apply_heuristic_in_file("outputs/tufano_predictions_raw_from_log_no_heuristics.txt")
    print("************** AFTER WITH HEURISTICS RESULT *******************")
    get_bleu_and_codebleu(
        "outputs/tufano_ground_truths_raw_from_log.txt", "outputs/tufano_predictions_raw_from_log_no_heuristics_heuristics_applied.txt"
    )
