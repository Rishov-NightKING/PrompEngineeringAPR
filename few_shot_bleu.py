from utils import (
    get_bleu_and_codebleu,
    transfer_content_to_another_file,
    apply_heuristic_in_file,
    write_list_to_file,
    get_EM_R4R,
)

if __name__ == "__main__":
    tufano_prediction_path = "few_shot_outputs/few_shot_tufano_predictions_raw_no_heuristic_0_1718.txt"
    apply_heuristic_in_file(tufano_prediction_path)
    tufano_prediction_formatted_path = "few_shot_outputs/few_shot_tufano_predictions_raw_no_heuristic_0_1718_formatted.txt"
    tufano_ground_truth_formatted_path = "few_shot_outputs/few_shot_tufano_ground_truths_raw_no_heuristic_0_1718.txt"

    get_bleu_and_codebleu(tufano_prediction_path, tufano_ground_truth_formatted_path)