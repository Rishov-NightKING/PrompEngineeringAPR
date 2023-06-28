import openai
from utils import (
    read_raw_tufano_dataset_from_csv,
    get_env_variable,
    get_few_shot_predictions_from_openai_and_write_to_file,
)

if __name__ == "__main__":
    openai.api_key = get_env_variable("OPENAI_API_KEY")
    OUTPUT_DIRECTORY = "few_shot_outputs"
    TUFANO_RAW_TRAIN_DATASET_FILE_PATH = "datasets/tufano/raw_train.csv"
    TUFANO_RAW_TEST_DATASET_FILE_PATH = "datasets/tufano/raw_test.csv"

    START_INDEX = 0
    END_INDEX = 6

    test_code_reviews, test_buggy_codes, test_target_codes = read_raw_tufano_dataset_from_csv(
        TUFANO_RAW_TEST_DATASET_FILE_PATH
    )
    train_code_reviews, train_buggy_codes, train_target_codes = read_raw_tufano_dataset_from_csv(
        TUFANO_RAW_TRAIN_DATASET_FILE_PATH
    )

    get_few_shot_predictions_from_openai_and_write_to_file(
        prediction_file_path=f"{OUTPUT_DIRECTORY}/few_shot_tufano_predictions_raw_no_heuristic.txt",
        ground_truth_path=f"{OUTPUT_DIRECTORY}/few_shot_tufano_ground_truths_raw_no_heuristic.txt",
        train_dataset=(train_code_reviews, train_buggy_codes, train_target_codes),
        test_dataset=(test_code_reviews, test_buggy_codes, test_target_codes),
        top_k=3
    )
