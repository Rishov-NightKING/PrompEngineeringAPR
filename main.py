import openai
from utils import (
    get_env_variable,
    read_raw_tufano_dataset_from_csv,
    get_predictions_from_openai_and_write_to_file,
    transfer_content_to_another_file,
)


if __name__ == "__main__":
    openai.api_key = get_env_variable("OPENAI_API_KEY")

    OUTPUT_DIRECTORY = "outputs"
    TUFANO_SOURCE_FILE_PATH = "datasets/tufano/test_CC_src.txt"
    TUFANO_TARGET_FILE_PATH = "datasets/tufano/test_CC_tgt.txt"
    TUFANO_RAW_DATASET_FILE_PATH = "datasets/tufano/raw_test.csv"
    R4R_SOURCE_FILE_PATH = "datasets/R4R/test_CC_src.txt"
    R4R_TARGET_FILE_PATH = "datasets/R4R/test_CC_tgt.txt"

    START_INDEX = 1500
    END_INDEX = 1719
    # code_reviews, buggy_codes, target_codes = read_dataset(
    #     dataset_name="R4R", source_file_path=R4R_SOURCE_FILE_PATH, target_file_path=R4R_TARGET_FILE_PATH
    # )

    code_reviews, buggy_codes, target_codes = read_raw_tufano_dataset_from_csv(TUFANO_RAW_DATASET_FILE_PATH)

    get_predictions_from_openai_and_write_to_file(
        f"{OUTPUT_DIRECTORY}/tufano_predictions_raw.txt",
        f"{OUTPUT_DIRECTORY}/tufano_ground_truths_raw.txt",
        code_reviews,
        buggy_codes,
        target_codes,
        START_INDEX,
        END_INDEX,
    )

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
