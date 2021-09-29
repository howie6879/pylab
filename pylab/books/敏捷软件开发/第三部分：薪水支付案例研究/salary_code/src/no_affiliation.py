#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""

from src.affiliation import Affiliation
from src.paycheck import Paycheck


class NoAffiliation(Affiliation):
    def calculate_deductions(self, paycheck: Paycheck):
        return 0
