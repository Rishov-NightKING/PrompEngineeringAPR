from utils import get_bleu_and_codebleu, transfer_content_to_another_file, apply_heuristic_in_file

if __name__ == "__main__":
    tufano_prediction_path = "outputs/edit_tufano_predictions_raw_no_heuristic_full.txt"
    tufano_prediction_formatted_path = "outputs/edit_tufano_predictions_raw_no_heuristic_full_formatted.txt"
    tufano_ground_truth_formatted_path = "outputs/edit_tufano_ground_truths_raw_no_heuristic_full.txt"
    tufano_edit_log_file = "logs/EDIT_LOGS_edit_tufano_predictions_raw_no_heuristic_full.txt"

    # transfer_content_to_another_file(keyword="response:",
    #                                  input_file=tufano_edit_log_file,
    #                                  output_file="outputs/edit_tufano_predictions_raw_no_heuristic_full.txt"
    #                                  )
    # transfer_content_to_another_file(keyword="target code:",
    #                                  input_file=tufano_edit_log_file,
    #                                  output_file="outputs/edit_tufano_ground_truths_raw_no_heuristic_full.txt"
    #                                  )

    # apply_heuristic_in_file(tufano_prediction_path)
    # get_bleu_and_codebleu(
    #     prediction_file_path=tufano_prediction_formatted_path, ground_truth_path=tufano_ground_truth_formatted_path
    # )

    r4r_edit_log_file = "logs/EDIT_LOGS_edit_r4r_predictions_raw_no_heuristic_0_2955.txt"
    # transfer_content_to_another_file(keyword="response:",
    #                                  input_file=r4r_edit_log_file,
    #                                  output_file="outputs/edit_r4r_predictions_raw_no_heuristic_full.txt"
    #                                  )
    # transfer_content_to_another_file(keyword="target code:",
    #                                  input_file=r4r_edit_log_file,
    #                                  output_file="outputs/edit_r4r_ground_truths_raw_no_heuristic_full.txt"
    #                                  )

    r4r_prediction_path = "outputs/edit_r4r_predictions_raw_no_heuristic_full.txt"
    # apply_heuristic_in_file(r4r_prediction_path)
    print("before heuristics")
    get_bleu_and_codebleu(
        prediction_file_path="outputs/r4r_predictions_raw_no_heuristic_0_2954_formatted.txt",
        ground_truth_path="outputs/r4r_ground_truths_raw_no_heuristic_0_2954_formatted.txt",
    )
    print("after heuristics")
    get_bleu_and_codebleu(
        prediction_file_path="outputs/r4r_predictions_raw_with_heuristic_0_2954_formatted.txt",
        ground_truth_path="outputs/r4r_ground_truths_raw_no_heuristic_0_2954_formatted.txt",
    )
