#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""
from src.change_classification_transaction import ChangeClassificationTransaction
from src.hourly_classification import HourlyClassification
from src.weekly_schedule import WeeklySchedule


class ChangeHourlyTransaction(ChangeClassificationTransaction):
    def __init__(self, emp_id, hourly_salary):
        super().__init__(emp_id)
        self.hourly_salary = hourly_salary

    def get_classification(self):
        return HourlyClassification(self.hourly_salary)

    def get_schedule(self):
        return WeeklySchedule(HourlyClassification.FRIDAY)
