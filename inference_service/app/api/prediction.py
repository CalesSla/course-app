from flask import abort, Blueprint, request
from pydantic import ValidationError

from schema.apartment import Apartment
from services import model_inference_service

bp = Blueprint("prediction", __name__, url_prefix="/pred")

@bp.get("/")
def get_prediction():

    try: 
        apartment_features = Apartment(**request.args)
    except ValidationError:
        abort(code=400, description="Bad input params")

    prediction = model_inference_service.predict(
        list(apartment_features.model_dump().values())
    )

    return {"prediction": prediction}
    

@bp.post("/")
def get_prediction_post():
    try: 
        apartment_features = Apartment(**request.json)
    except ValidationError:
        abort(code=400, description="Bad input params")

    prediction = model_inference_service.predict(
        list(apartment_features.model_dump().values())
    )

    return {"prediction": prediction}