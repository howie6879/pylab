#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""

from src.change_employee_transaction import ChangeEmployeeTransaction, Employee


class ChangeNameTransaction(ChangeEmployeeTransaction):
    def __init__(self, emp_id: int, name: str):
        super().__init__(emp_id=emp_id)
        self.name = name

    def change(self, employee: Employee):
        employee.name = self.name
