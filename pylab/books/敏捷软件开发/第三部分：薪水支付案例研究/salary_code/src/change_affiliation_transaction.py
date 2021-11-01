#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""
from src.change_employee_transaction import ChangeEmployeeTransaction
from src.employee import Employee


class ChangeAffiliationTransaction(ChangeEmployeeTransaction):
    def __init__(self, emp_id):
        super().__init__(emp_id=emp_id)

    def change(self, employee: Employee):
        self.record_membership(employee)
        employee.affiliation = self.get_affiliation()

    def get_affiliation(self):
        raise NotImplementedError

    def record_membership(self, employee: Employee):
        raise NotImplementedError
