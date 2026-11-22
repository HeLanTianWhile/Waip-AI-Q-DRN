# 版本号2.6.3.0
from math import tanh


def main(NeuralConfigDatabase, SynapsesDatabase, NeuralStateDatabase, Input):

    print(Input)

    pending_tasks = []
    for i in NeuralStateDatabase:
        if NeuralStateDatabase[i]["y"] != "NOT Activate":
            pending_tasks.append(i)

    task_dispatch_list = {}
    if pending_tasks:
        for i in pending_tasks:
            for j in SynapsesDatabase:
                if i in SynapsesDatabase[j]["Connection Target"]:
                    Transmitted_Object = next(x for x in SynapsesDatabase[j]["Connection Target"] if x != i)
                    Transmitted_Number = SynapsesDatabase[j]["Weight"] * NeuralStateDatabase[i]["y"]
                    if Transmitted_Object in task_dispatch_list:
                        task_dispatch_list[Transmitted_Object] += Transmitted_Number
                    else:
                        task_dispatch_list[Transmitted_Object] = Transmitted_Number
    
    for i in NeuralConfigDatabase:
        if ("Input" in NeuralConfigDatabase[i]["IP"]) and (Input == NeuralConfigDatabase[i]["IP"][1]) and (Input != "<blank>"):
            if i in task_dispatch_list:
                task_dispatch_list[i] += 1
            else:
                task_dispatch_list[i] = 1

    pending_changes = {}
    for i in task_dispatch_list:
        pending_changes[i] = tanh(task_dispatch_list[i] - NeuralConfigDatabase[i]["bias"])

    for i in pending_tasks:
        if i not in pending_changes:
            pending_changes[i] = "NOT Activate"

    for i in pending_changes:
        NeuralStateDatabase[i]["y"] = pending_changes[i]

    
    return NeuralConfigDatabase, NeuralStateDatabase, SynapsesDatabase
