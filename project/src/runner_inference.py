from model.model_inference import ModelInferenceService
from loguru import logger


@logger.catch
def main():
    logger.info("running the application...")
    ml_svc = ModelInferenceService()
    ml_svc.load_model()
    
    feature_values = {
        "area": 85,
        "constraction_year": 2015,
        "bedrooms": 2,
        "garden_area": 20,
        "balcony_present": 1,
        "parking_present": 1,
        "furnished": 0,
        "garage_present": 0,
        "storage_present": 1,
    }
    pred = ml_svc.predict(list(feature_values.values()))
    logger.info(f"prediction = {pred}")


if __name__ == "__main__":
    main()
