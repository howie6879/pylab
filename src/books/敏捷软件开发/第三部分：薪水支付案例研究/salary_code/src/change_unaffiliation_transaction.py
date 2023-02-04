#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/22.
"""
from src.change_affiliation_transaction import ChangeAffiliationTransaction
from src.employee import Employee
from src.no_affiliation import NoAffiliation
from src.payroll_database import PayrollDatabase
from src.union_affiliation import UnionAffiliation


class ChangeUnaffiliatedTransaction(ChangeAffiliationTransaction):
    def __init__(self, emp_id: int):
        super().__init__(emp_id=emp_id)

    def get_affiliation(self):
        return NoAffiliation()

    def record_membership(self, employee: Employee):
        affiliation = employee.affiliation

        if isinstance(affiliation, UnionAffiliation):
            member_id = affiliation.member_id
            PayrollDatabase.remove_union_member(member_id)
