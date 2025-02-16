import unittest
from .car_class import Car


class TestCar(unittest.TestCase):
    def test_car(self):
        make = "Toyota"
        model = "Corolla"
        year = 2020
        num_seats = 5
        fuel_capacity = 50
        fuel_level = 17
        efficiency = 10

        car = Car(make, model, year, num_seats, fuel_capacity, efficiency)
        range_left = car.calc_range(fuel_level)

        assert range_left <= fuel_capacity * efficiency, "Range is too high!"
