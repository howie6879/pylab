#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""

from src.paycheck import Paycheck
from src.payment_method import PaymentMethod


class HoldMethod(PaymentMethod):
    def pay(self, paycheck: Paycheck):
        paycheck.disposition = "Hold"
