#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""

from src.add_employee_transaction import AddEmployeeTransaction
from src.hourly_classification import HourlyClassification
from src.weekly_schedule import WeeklySchedule


class AddHourlyEmployee(AddEmployeeTransaction):
    def __init__(self, emp_id: int, name: str, address: str, hour_salary: float):
        super().__init__(emp_id, name, address)
        self.hour_salary = hour_salary

    def make_classification(self):
        return HourlyClassification(self.hour_salary)

    def make_schedule(self):
        return WeeklySchedule(HourlyClassification.FRIDAY)
