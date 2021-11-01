#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""

from src.change_affiliation_transaction import ChangeAffiliationTransaction
from src.employee import Employee
from src.payroll_database import PayrollDatabase
from src.union_affiliation import UnionAffiliation


class ChangeMemberTransaction(ChangeAffiliationTransaction):
    def __init__(self, emp_id: int, member_id: int, dues: float):
        super().__init__(emp_id)
        self.member_id = member_id
        self.dues = dues

    def get_affiliation(self):
        return UnionAffiliation(self.member_id, self.dues)

    def record_membership(self, employee: Employee):
        PayrollDatabase.add_union_member(self.member_id, employee)
