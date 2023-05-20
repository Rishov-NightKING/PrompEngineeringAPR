import openai
from utils import get_env_variable, read_raw_tufano_dataset_from_csv, get_predictions_from_openapi_and_write_to_file


if __name__ == "__main__":
    openai.api_key = get_env_variable("OPENAI_API_KEY")

    TUFANO_SOURCE_FILE_PATH = "datasets/tufano/test_CC_src.txt"
    TUFANO_TARGET_FILE_PATH = "datasets/tufano/test_CC_tgt.txt"
    TUFANO_RAW_DATASET_FILE_PATH = "datasets/tufano/raw_test.csv"

    # code_reviews, buggy_codes, target_codes = read_dataset(
    #     dataset_name="tufano", source_file_path=TUFANO_SOURCE_FILE_PATH, target_file_path=TUFANO_TARGET_FILE_PATH
    # )

    code_reviews, buggy_codes, target_codes = read_raw_tufano_dataset_from_csv(TUFANO_RAW_DATASET_FILE_PATH)
    get_predictions_from_openapi_and_write_to_file("predictions.txt", code_reviews, buggy_codes, target_codes)
