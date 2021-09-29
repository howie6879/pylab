#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""
from src.employee import Employee
from src.payroll_database import PayrollDatabase
from src.transaction import Transaction


class ChangeEmployeeTransaction(Transaction):
    def __init__(self, emp_id: int):
        self.emp_id = emp_id

    def execute(self):
        e = PayrollDatabase.get_employee(self.emp_id)
        if e is not None:
            self.change(e)

    def change(self, employee: Employee):
        raise NotImplementedError
