#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""
import datetime

from src.affiliation import Affiliation
from src.no_affiliation import NoAffiliation
from src.paycheck import Paycheck
from src.payment_classification import PaymentClassification
from src.payment_method import PaymentMethod
from src.payment_schedule import PaymentSchedule


class Employee:
    def __init__(self, emp_id, name, address):
        self.emp_id = emp_id
        self.name = name
        self.address = address

        self.affiliation: Affiliation = NoAffiliation()
        self.classification: PaymentClassification = None
        self.method: PaymentMethod = None
        self.schedule: PaymentSchedule = None

    def is_pay_date(self, date: datetime.date):
        return self.schedule.is_pay_day(date)

    def get_pay_period_start_date(self, pay_date: datetime):
        return self.schedule.get_pay_period_start_date(pay_date)

    def payday(self, paycheck: Paycheck):
        # 总收入
        gross_pay = self.classification.calculate_pay(paycheck)
        # 扣除部分
        deductions = self.affiliation.calculate_deductions(paycheck)
        # 最终结果
        net_pay = gross_pay - deductions

        paycheck.gross_pay = gross_pay
        paycheck.deductions = deductions
        paycheck.net_pay = net_pay

        # 付款方式
        self.method.pay(paycheck)
