#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/21.
"""
import datetime

from src.transaction import Transaction
from src.payroll_database import PayrollDatabase
from src.service_charge import ServiceCharge
from src.union_affiliation import UnionAffiliation


class ServiceChargeTransaction(Transaction):
    def __init__(self, member_id: int, date: datetime.date, charge: float):
        self.member_id = member_id
        self.date = date
        self.charge = charge

    def execute(self):
        e = PayrollDatabase.get_union_member(self.member_id)
        if e is not None:
            ua = None
            if isinstance(e.affiliation, UnionAffiliation):
                ua = e.affiliation
            if ua is not None:
                ua.add_service_charge(ServiceCharge(self.date, self.charge))
            else:
                raise ValueError(
                    "Tries to add service charge to union member without a union affiliation"
                )
        else:
            raise ValueError("No such union member")
