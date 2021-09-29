#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""
import typing

from src.employee import Employee


class PayrollDatabase:
    employees = {}
    member_employees = {}

    @classmethod
    def add_employee(cls, emp_id: int, employee: Employee):
        cls.employees[emp_id] = employee

    @classmethod
    def delete_employee(cls, emp_id: int):
        cls.employees.pop(emp_id)

    @classmethod
    def get_employee(cls, emp_id: int) -> typing.Union[Employee, None]:
        try:
            return cls.employees[emp_id]
        except:
            return None

    @classmethod
    def get_all_employee_ids(cls):
        return list(cls.employees.keys())

    @classmethod
    def add_union_member(cls, member_id: int, employee: Employee):
        cls.member_employees[member_id] = employee.emp_id

    @classmethod
    def get_union_member(cls, member_id: int) -> typing.Union[Employee, None]:
        try:
            return cls.employees[cls.member_employees[member_id]]
        except:
            return None

    @classmethod
    def remove_union_member(cls, member_id):
        cls.member_employees.pop(member_id)
