#!/usr/bin/env python
"""
 Created by howie.hu at 2019/4/19.
"""

from abc import ABC, abstractmethod


class Transaction(ABC):
    """
    COMMAND 设计模式，此接口代表事物
    """

    @abstractmethod
    def execute(self):
        pass
