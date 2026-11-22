import random


def data_into(input=74884, output=4, memory=500, Normal=2000):
    NeuralConfigDatabase = {}
    SynapsesDatabase = {}
    NeuralStateDatabase = {}
    index = 0
    SynapsesIndex = 0
    input_list = []
    output_list = []
    memory_list = []
    normal_list = []
    for i in range(input):
        index += 1
        s = str(index).zfill(4)  
        NeuralConfigDatabase[s] = {"bias": random.uniform(-1.0, 1.0), "IP": ["Input", i]}
        NeuralStateDatabase[s] = {"y": "NOT Activate"}
        input_list.append(s)
    for i in range(output):
        index += 1
        s = str(index).zfill(4)  
        NeuralConfigDatabase[s] = {"bias": random.uniform(-1.0, 1.0), "IP": ["Output", i]}
        NeuralStateDatabase[s] = {"y": "NOT Activate"}
        output_list.append(s)
    for i in range(memory):
        index += 1
        s = str(index).zfill(4)  
        NeuralConfigDatabase[s] = {"bias": random.uniform(-1.0, 1.0), "IP": ["Memory", i]}
        NeuralStateDatabase[s] = {"y": "NOT Activate"}
        memory_list.append(s)
    for i in range(Normal):
        index += 1
        s = str(index).zfill(4)  
        NeuralConfigDatabase[s] = {"bias": random.uniform(-1.0, 1.0), "IP": ["Normal", i]}
        NeuralStateDatabase[s] = {"y": "NOT Activate"}
        normal_list.append(s)

    for i in range(len(memory_list)):
        s = str(SynapsesIndex).zfill(4)
        if i == memory - 1:
            SynapsesDatabase[s] = {"Weight": random.uniform(-1.0, 1.0), "Connection Target": [memory_list[i], memory_list[0]]}
        else:
            SynapsesDatabase[s] = {"Weight": random.uniform(-1.0, 1.0), "Connection Target": [memory_list[i], memory_list[i+1]]}
        SynapsesIndex += 1
    
    
    Hidden_list = memory_list + normal_list
    for i in input_list:
        pending_connections = random.sample(Hidden_list, random.randint(1, 5))
        for j in pending_connections:
            s = str(SynapsesIndex).zfill(4)
            SynapsesDatabase[s] = {"Weight": random.uniform(-1.0, 1.0), "Connection Target": [i, j]}
            SynapsesIndex += 1
    connected_output_neurons_list = []
    for i in normal_list:
        candidates = [x for x in Hidden_list if x != i]
        pending_connections = random.sample(candidates, random.randint(1, 5))
        for j in pending_connections:
            s = str(SynapsesIndex).zfill(4)
            SynapsesDatabase[s] = {"Weight": random.uniform(-1.0, 1.0), "Connection Target": [i, j]}
            SynapsesIndex += 1
        pending_connections_output = random.sample(output_list, random.randint(0, 1))
        for j in pending_connections_output:
            s = str(SynapsesIndex).zfill(4)
            SynapsesDatabase[s] = {"Weight": random.uniform(-1.0, 1.0), "Connection Target": [i, j]}
            SynapsesIndex += 1
            connected_output_neurons_list.append(j)

    unconnected_output_neurons_list = [x for x in output_list if x not in connected_output_neurons_list]
    for i in unconnected_output_neurons_list:
        pending_connections = random.sample(Hidden_list, random.randint(1, 5))
        for j in pending_connections:
            s = str(SynapsesIndex).zfill(4)
            SynapsesDatabase[s] = {"Weight": random.uniform(-1.0, 1.0), "Connection Target": [i, j]}
            SynapsesIndex += 1

    return NeuralConfigDatabase, SynapsesDatabase, NeuralStateDatabase
