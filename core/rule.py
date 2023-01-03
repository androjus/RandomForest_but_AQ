"""
rule.py
Author: Arkadiusz Nowacki
"""
class Rule:

    def __init__(self, n_columns: int, result: list):
        self.conditions = [["?"] for x in range(n_columns)] 
        self.result = result

    def set_conditions(self, conditions: list):
        self.conditions = conditions
    
    def print_rule(self) -> None:
        print(f"{self.conditions} -> {self.result}")

    def size(self):
        return len(self.conditions)

    def get_reslut(self):
        return self.result

    def cover(self, conditions: list) -> bool:
        if len(conditions) != len(self.conditions):
            print("Wrong size of complex!")
            return

        for iter in range(len(conditions)):
            if self.conditions[iter] == ["?"] or not self.conditions[iter]:
                continue
            if conditions[iter] not in self.conditions[iter]:
                return False
        return True