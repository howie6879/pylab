#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""
from src.add_employee_transaction import AddEmployeeTransaction
from src.monthly_schedule import MonthlySchedule
from src.salaried_classification import SalariedClassification


class AddSalariedEmployee(AddEmployeeTransaction):
    def __init__(self, emp_id: int, name: str, address: str, salary: float):
        super().__init__(emp_id, name, address)
        self.salary = salary

    def make_classification(self):
        return SalariedClassification(self.salary)

    def make_schedule(self):
        return MonthlySchedule()
