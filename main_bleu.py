from utils import (
    get_bleu_and_codebleu,
    transfer_content_to_another_file,
    apply_heuristic_in_file
)

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

    apply_heuristic_in_file(tufano_prediction_path)
    get_bleu_and_codebleu(prediction_file_path=tufano_prediction_formatted_path,
                          ground_truth_path=tufano_ground_truth_formatted_path)