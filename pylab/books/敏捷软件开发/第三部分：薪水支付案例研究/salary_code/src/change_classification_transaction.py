#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""

from src.change_employee_transaction import ChangeEmployeeTransaction, Employee


class ChangeClassificationTransaction(ChangeEmployeeTransaction):
    def __init__(self, emp_id: int):
        super().__init__(emp_id=emp_id)

    def change(self, employee: Employee):
        employee.classification = self.get_classification()
        employee.schedule = self.get_schedule()

    def get_classification(self):
        raise NotImplementedError

    def get_schedule(self):
        raise NotImplementedError
