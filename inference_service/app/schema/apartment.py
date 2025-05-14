from pydantic import BaseModel

class Apartment(BaseModel):
    area: int
    constraction_year: int
    bedrooms: int
    garden_area: int
    balcony_present: int
    parking_present: int
    furnished: int
    garage_present: int
    storage_present: int