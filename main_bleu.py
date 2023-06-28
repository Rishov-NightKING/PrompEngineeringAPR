from utils import (
    get_bleu_and_codebleu,
    transfer_content_to_another_file,
    apply_heuristic_in_file,
    write_list_to_file,
    get_EM_R4R,
)

if __name__ == "__main__":
    #############################TUFANO_EDIT##################################
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

    #############################R4R_GPT##################################
    # r4r_prediction_path = "outputs/edit_r4r_predictions_raw_no_heuristic_full.txt"
    # # apply_heuristic_in_file(r4r_prediction_path)
    # print("before heuristics")
    # get_bleu_and_codebleu(
    #     prediction_file_path="outputs/r4r_predictions_raw_no_heuristic_0_2954_formatted.txt",
    #     ground_truth_path="outputs/r4r_ground_truths_raw_no_heuristic_0_2954_formatted.txt",
    # )
    # print("after heuristics")
    # get_bleu_and_codebleu(
    #     prediction_file_path="outputs/r4r_predictions_raw_with_heuristic_0_2954_formatted.txt",
    #     ground_truth_path="outputs/r4r_ground_truths_raw_no_heuristic_0_2954_formatted.txt",
    # )

    #############################R4R_EDIT##################################
    r4r_prediction_path = "outputs/edit_r4r_predictions_raw_no_heuristic_full.txt"
    r4r_ground_truths_path = "outputs/edit_r4r_ground_truths_raw_no_heuristic_full.txt"
    r4r_prediction_formatted_path = "outputs/edit_r4r_predictions_raw_no_heuristic_full_formatted.txt"
    r4r_ground_truth_formatted_path = "outputs/edit_r4r_ground_truths_raw_no_heuristic_full_formatted.txt"
    r4r_edit_log_file = "logs/EDIT_LOGS_edit_r4r_predictions_raw_no_heuristic_0_2955.txt"

    # transfer_content_to_another_file(keyword="response:",
    #                                  input_file=r4r_edit_log_file,
    #                                  output_file=r4r_prediction_path
    #                                  )
    # transfer_content_to_another_file(keyword="target code:",
    #                                  input_file=r4r_edit_log_file,
    #                                  output_file=r4r_ground_truths_path
    #                                  )

    # apply_heuristic_in_file(r4r_prediction_path)
    # apply_heuristic_in_file(r4r_ground_truths_path)

    print("before heuristics")
    get_bleu_and_codebleu(
        prediction_file_path=r4r_prediction_formatted_path,
        ground_truth_path=r4r_ground_truth_formatted_path,
    )
    # print("after heuristics")
    # get_bleu_and_codebleu(
    #     prediction_file_path=r4r_prediction_formatted_path,
    #     ground_truth_path=r4r_ground_truth_formatted_path,
    # )

    ##################################### R4R_EDIT_SAMPLE ############################
    r4r_ground_truths_gpt_em_formatted_file = "outputs/r4r_ground_truth_paths_modified_for_EM_formatted.txt"
    r4r_ground_truths_edit_em_formatted_file = "outputs/edit_r4r_ground_truth_paths_modified_for_EM_formatted.txt"

    # with open(r4r_edit_log_file, "r", encoding="UTF-8") as r4r_edit_log, open(r4r_ground_truths_gpt_em_formatted_file, "r", encoding="UTF-8") as r4r_ground_truths_gpt_em_formatted:
    #     r4r_edit_log_lines = r4r_edit_log.readlines()
    #     r4r_ground_truths_gpt_em_formatted_lines = r4r_ground_truths_gpt_em_formatted.readlines()

    # r4r_edit_samples = []
    # r4r_ground_truths_edit_samples = []
    # for line in r4r_edit_log_lines:
    #     if line.startswith("sample: "):
    #         sample_no = int(line.split("sample: ")[1])
    #         r4r_ground_truths_edit_sample = r4r_ground_truths_gpt_em_formatted_lines[sample_no].strip()

    #         r4r_edit_samples.append(sample_no)
    #         r4r_ground_truths_edit_samples.append(r4r_ground_truths_edit_sample)

    # # print(r4r_edit_samples)
    # # print(len(r4r_edit_samples))

    # write_list_to_file(r4r_ground_truths_edit_em_formatted_file, r4r_ground_truths_edit_samples)

    # get_EM_R4R(r4r_ground_truths_edit_em_formatted_file, r4r_prediction_formatted_path)

    with open(r4r_ground_truths_edit_em_formatted_file, "r", encoding="UTF-8") as f1, open(
        r4r_prediction_formatted_path, "r", encoding="UTF-8"
    ) as f2:
        ground_truths = f1.readlines()
        preds = f2.readlines()

        start_focus_tag = "< |startfocus| >"
        end_focus_tag = "< |endfocus| >"

        exact_matches = []
        del_matches = []
        possible_duplicates = []
        matches_r_equal_p = []
        no_focus_count = 0
        no_focus_in_matches = []
        for i, (ground_truth, pred) in enumerate(zip(ground_truths, preds)):
            ground_truth = ground_truth.strip()
            pred = pred.strip()

            try:
                pred_focus = pred.split(start_focus_tag)[1]
                pred_focus = pred_focus.split(end_focus_tag)[0]
            except Exception as e:
                # print(f"something happend at sample: {i}, exception: {e}")
                no_focus_count += 1
                if ground_truth.strip() in pred.strip():
                    no_focus_in_matches.append(i)
                continue

            del_token = "< |del| >"

            if ground_truth.startswith(del_token):
                if pred_focus == " ":
                    exact_matches.append(i)
            else:
                if ground_truth.strip() in pred_focus.strip():
                    if ground_truth.strip() == pred_focus.strip():
                        exact_matches.append(i)
                    else:
                        possible_duplicates.append(i)

        print(f"exact matches: {exact_matches}, len: {len(exact_matches)}")
        print(f"possible_duplicates matches: {possible_duplicates}, len: {len(possible_duplicates)}")
        print(f"no_focus_count: {no_focus_count}")
        print(f"no_focus_in_matches: {no_focus_in_matches}, len:{len(no_focus_in_matches)}")
        print(f"EM: {(len(exact_matches) + len(no_focus_in_matches))/ len(ground_truths)}")
