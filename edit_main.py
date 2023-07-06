from utils import get_EM, get_bleu_and_codebleu

OUTPUT_DIRECTORY = "edit_outputs"

get_EM(
    f"{OUTPUT_DIRECTORY}/edit_r4r_predictions_raw.txt",
    f"{OUTPUT_DIRECTORY}/edit_r4r_ground_truths_raw_target_only.txt",
    "R4R"
)

get_bleu_and_codebleu(
    f"{OUTPUT_DIRECTORY}/edit_r4r_predictions_raw.txt",
    f"{OUTPUT_DIRECTORY}/edit_r4r_ground_truths_raw.txt"
)

get_EM(
    f"{OUTPUT_DIRECTORY}/edit_tufano_predictions_raw.txt",
    f"{OUTPUT_DIRECTORY}/edit_tufano_ground_truths_raw.txt",
    "tufano"
)

get_bleu_and_codebleu(
    f"{OUTPUT_DIRECTORY}/edit_tufano_predictions_raw.txt",
    f"{OUTPUT_DIRECTORY}/edit_tufano_ground_truths_raw.txt"
)