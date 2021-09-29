#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/20.
"""

from src.transaction import Transaction
from src.payroll_database import PayrollDatabase


class DeleteEmployeeTransaction(Transaction):
    def __init__(self, emp_id: int):
        self._emp_id = emp_id

    def execute(self):
        PayrollDatabase.delete_employee(emp_id=self._emp_id)
