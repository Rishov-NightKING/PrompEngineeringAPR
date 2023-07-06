from utils import (
    get_bleu_and_codebleu,
    transfer_content_to_another_file,
    apply_heuristic_in_file,
    write_list_to_file,
    get_EM_R4R,
    heuristic_adjust_spaces
)

if __name__ == "__main__":
    tufano_prediction_path = "few_shot_outputs/few_shot_tufano_predictions_raw_no_heuristic_0_1718.txt"
    apply_heuristic_in_file(tufano_prediction_path)
    tufano_prediction_formatted_path = "few_shot_outputs/few_shot_tufano_predictions_raw_no_heuristic_0_1718_formatted.txt"
    tufano_prediction_after_heuristics_formatted_path = "few_shot_outputs/few_shot_tufano_predictions_raw_no_heuristic_0_1718_heuristics_applied.txt"
    tufano_ground_truth_formatted_path = "few_shot_outputs/few_shot_tufano_ground_truths_raw_no_heuristic_0_1718.txt"

    # print(f"---------------TUFANO BEFORE HEURISTICS------------------")
    # get_bleu_and_codebleu(tufano_prediction_path, tufano_ground_truth_formatted_path)

    # print(f"---------------TUFANO AFTER HEURISTICS------------------")
    # get_bleu_and_codebleu(tufano_prediction_after_heuristics_formatted_path, tufano_ground_truth_formatted_path)


    ##################### R4R  BLEU CODEBLEU ##########################

    # transferring content
    r4r_few_shot_log_file = "logs/FEW_SHOT_LOGS_few_shot_r4r_predictions_raw_no_heuristic_full.txt"
    r4r_prediction_path = "few_shot_outputs/few_shot_r4r_predictions_raw_no_heuristic_full.txt"
    r4r_prediction_formatted_path = "few_shot_outputs/few_shot_r4r_predictions_raw.txt"
    
    r4r_pred_lines = open(r4r_prediction_path, "r", encoding="UTF-8").readlines()
    # write_list_to_file(r4r_prediction_formatted_path, [heuristic_adjust_spaces(line) for line in r4r_pred_lines])

    # transfer_content_to_another_file("response: ", r4r_few_shot_log_file, r4r_prediction_path)
    # apply_heuristic_in_file(r4r_prediction_path)
    
    r4r_prediction_after_heuristics_formatted_path = "few_shot_outputs/few_shot_r4r_predictions_raw_applied_heuristics.txt"
    
    r4r_ground_truth_path = "outputs/r4r_ground_truths_raw_no_heuristic_0_2954.txt"
    r4r_ground_truth_formatted_path = "outputs/r4r_ground_truths_raw_no_heuristic_0_2954_formatted.txt"
    r4r_ground_truth_for_em_formatted_path = "outputs/r4r_ground_truth_paths_modified_for_EM_formatted.txt"
    
    # checking score
    print(f"---------------R4R BEFORE HEURISTICS------------------")
    get_bleu_and_codebleu(r4r_prediction_path, r4r_ground_truth_path)

    print(f"---------------R4R BEFORE HEURISTICS------------------")
    get_bleu_and_codebleu(r4r_prediction_formatted_path, r4r_ground_truth_formatted_path)

    print(f"---------------R4R AFTER HEURISTICS------------------")
    get_bleu_and_codebleu(r4r_prediction_after_heuristics_formatted_path, r4r_ground_truth_formatted_path)
    
    print(f"-------------R4R EM BEFORE HEURISTIC-----------------")
    get_EM_R4R(r4r_ground_truth_path, r4r_prediction_path)
    
    print(f"-------------R4R EM BEFORE HEURISTIC-----------------")
    get_EM_R4R(r4r_ground_truth_formatted_path, r4r_prediction_formatted_path)
    print(f"-------------R4R EM AFTER HEURISTIC-----------------")
    get_EM_R4R(r4r_ground_truth_for_em_formatted_path, r4r_prediction_after_heuristics_formatted_path)