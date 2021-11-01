#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""
import datetime

from src.paycheck import Paycheck
from src.payroll_database import PayrollDatabase
from src.transaction import Transaction


class PaydayTransaction(Transaction):
    def __init__(self, date: datetime.date):
        self.date = date
        self.paycheck_map = {}

    def execute(self):
        emp_ids = PayrollDatabase.get_all_employee_ids()
        for emp_id in emp_ids:
            employee = PayrollDatabase.get_employee(emp_id)
            if employee.is_pay_date(self.date):
                paycheck = Paycheck(
                    employee.get_pay_period_start_date(self.date), self.date
                )
                self.paycheck_map[emp_id] = paycheck
                employee.payday(paycheck)

    def get_paycheck(self, emp_id):
        return self.paycheck_map.get(emp_id)
