import numpy as np
from pathlib import Path
import pickle
import copy

from core.rule import Rule

class AQ:

    def __init__(self):
        self.rules = []

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:

        while train_x.shape[0] and train_y.shape[0]: 
            rule = self.__specialization(train_x=train_x, train_y=train_y) 
            self.rules.append(rule) #adding a rule resulting from specialization

            covered = []
            for idx, value in enumerate(train_x): #check covered complexes
                if(rule.cover(value)):
                    covered.append(idx)

            train_x = np.delete(train_x, covered, axis=0) #remove covered complexes
            train_y = np.delete(train_y, covered, axis=0)

    def __specialization(self, train_x: np.ndarray, train_y: np.ndarray) -> Rule:

        pos_krenel_conditions = train_x[0] #choosing a positive kernel
        pos_kerenl_result = train_y[0]

        pos_conditions = train_x[np.where(train_y == pos_kerenl_result)[0]] #creating a list of positive and negative complexes
        neg_conditions = train_x[np.where(train_y != pos_kerenl_result)[0]]

        rules = [Rule(n_columns = len(pos_krenel_conditions), result=pos_kerenl_result)] #initialization of the rule list

        while neg_conditions.shape[0]:

            neg_condition = neg_conditions[0] #choosing a negative kernel

            if((pos_krenel_conditions == neg_condition).all()):  #if all values are the same - remove,
                neg_conditions = np.delete(neg_conditions, 0, axis=0) #it is impossible to perform specialization for the same negative and positive kernel
                continue
            
            for _, rule in enumerate(rules):
                if(rule.cover(neg_condition)): #checking if the rule covers the negative kernel
                    rules.remove(rule)
                    for idx, value in enumerate(neg_condition):
                        if(pos_krenel_conditions[idx] != value): 
                            new_rule = copy.deepcopy(rule)
                            if new_rule.conditions[idx] == ["?"]:
                                new_conditions = np.unique(train_x[:, idx]).tolist() #selecting unique values for a column
                                new_rule.conditions[idx] = new_conditions
                            if value in new_rule.conditions[idx]:
                                new_rule.conditions[idx].remove(value) #removal of rule found in negative kernel
                            rules.append(new_rule)
                            
            rules = self.__remove_non_general(rules=rules) #remove not general, select top n rule and delete covered complexes
            rules = self.__top_n(pos=pos_conditions, neg=neg_conditions, rules=rules)
            neg_conditions = self.__remove_covered(rules=rules, conditions=neg_conditions)
        
        result = Rule(n_columns=len(rules[0].conditions), result=rules[0].get_reslut()) #get first rule and return as specialization result
        result.set_conditions(conditions=rules[0].conditions)
        return result

    def __remove_covered(self, rules: list, conditions: np.array) -> np.array:
        conditions_list = conditions.tolist()
        for neg in conditions_list:
            for rule in rules:
                if(not rule.cover(neg)):
                    if neg in conditions_list:
                        conditions_list.remove(neg)
        return np.array(conditions_list)

    def __remove_non_general(self, rules: list) -> list:
        for _, rule in enumerate(rules):
            rules_rest = rules.copy()
            rules_rest.remove(rule)
            for rule_rest in rules_rest:
                more_general = False
                if any(rule.conditions[idx] == ["?"] and condition != ["?"] for idx, condition in enumerate(rule_rest.conditions)):
                    continue #if rule A has ? and rule B doesn;t, then A is more general
                for idx, con in enumerate(rule_rest.conditions):
                    if not all(value in con for value in rule.conditions[idx]): #if rule A has more values than rule B, then A is more general
                        more_general = True
                        break
                if more_general:
                    continue
                rules.remove(rule)
                break
        return rules

    def __top_n(self, rules: list, pos: np.array, neg: np.array, num_top: int = 2) -> list:
        if len(rules) <= 2:
            return rules

        scores = []
        for rule in rules:
            rule_score = 0

            for single_pos in pos:  #if the rule covers positive complexes, 1 point is added, if not -1
                if(rule.cover(single_pos)):
                    rule_score += 1
                else:
                    rule_score -= 1

            for single_neg in neg: #if the rule not covers negative complexes, 1 point is added, if not -1
                if(rule.cover(single_neg)):
                    rule_score -= 1
                else:
                    rule_score += 1
            scores.append(rule_score)

        rules_new = []
        for _ in range(num_top):
            best_ind = len(scores) - 1 - scores[::-1].index(max(scores)) #selection of the added last, if there are two same values
            rules_new.append(rules[best_ind])
            del scores[best_ind]
            del rules[best_ind]
        
        return rules_new
    
    def predict(self, complex: list) -> np.ndarray:
        result = None

        if(not self.rules):
            print("The rule set is empty!")
            return
        
        for rule in self.rules:
            if rule.size() != len(complex):
                print("Wrong size of complex!")
                return
            
            if rule.cover(complex):
                result = rule.get_reslut()
                break

        return result

    def save_rules(self, path: Path) -> None:
        with open(path, 'wb') as handle:
            pickle.dump(self.rules, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_rules(self, path: Path) -> None:
        with open(path, 'rb') as handle:
            self.rules = pickle.load(handle)