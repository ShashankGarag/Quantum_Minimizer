import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import XGate, ZGate
from qiskit import Aer
from qiskit import transpile
import numpy as np
import pickle
from timeit import Timer

decision_space = [241,708,647,179,1023,1020,1021,1022]


class QuantumMinimizer:

    # Initialize all the main values of the Quantum Circuit
    def __init__(self, inputs, outputs, search_space):
        self.inputs = inputs
        self.outputs = outputs
        self.search_space = search_space

    # Create the state preparation operator
    def state_preparation(self):
        # Create Circuit
        register = QuantumRegister(self.inputs + self.outputs)
        hadamards = QuantumCircuit(register)
        hadamards.h(list(range(self.inputs)))
        hadamards.x(self.inputs + self.outputs - 1)
        hadamards.h(self.inputs + self.outputs - 1)
        return hadamards

    # Create the U gate
    def u_gate(self):
        register = QuantumRegister(self.inputs + self.outputs)
        initialize = QuantumCircuit(register)
        custom_gate = XGate().control(self.inputs)

        # Create a list of binary values from the decision space (CHANGE TO MANUAL)
        binary_decision_space = []
        for x in self.search_space:
            s = format(x, "0" + str(self.outputs - 1) + "b")
            binary_decision_space.append(s)

        # Create list of binary indexes for each list value (CHANGE TO MANUAL)
        binary_indexes = []
        regular_indexes = []
        length = range(len(binary_decision_space))
        for index in length:
            bin_index = format(index, "0" + str(self.inputs) + "b")
            binary_indexes.append(bin_index)
        for i in length:
            regular_indexes.append(i)

        # Number of control qubits
        state_qubits = []
        for i in range(len(binary_indexes[0])):
            state_qubits.append(i)
        state_qubits.append(6)

        # Create circuit
        for (i, value) in zip(regular_indexes, binary_indexes):
            for digit_index in range(len(binary_decision_space[i])):
                if binary_decision_space[i][digit_index] == '1':
                    state_qubits[-1] = self.inputs + digit_index
                    for index in range(len(value)):
                        if value[index] == '0':
                            initialize.x(register[index])
                    initialize.append(custom_gate, state_qubits)
                    for index in range(len(value)):
                        if value[index] == '0':
                            initialize.x(register[index])
                    initialize.barrier()
        return initialize

    # Create the Diffusion Operator
    def diffuser(self):
        inputs = []
        for i in range(self.inputs):
            inputs.append(i)
        register = QuantumRegister(self.inputs + self.outputs)
        diffuser = QuantumCircuit(register)
        diffuser.h(range(self.inputs))
        diffuser.x(range(self.inputs))
        diffuser.h(self.inputs - 1)
        custom_gate = ZGate().control(self.inputs - 1)
        diffuser.append(custom_gate, inputs)
        diffuser.h(self.inputs - 1)
        diffuser.x(range(self.inputs))
        diffuser.h(range(self.inputs))
        return diffuser

    def save_objects(self):
        with open('state_prep.pkl', 'wb') as file:
            pickle.dump(self.state_preparation(), file)
        with open('u_gate.pkl', 'wb') as file:
            pickle.dump(self.u_gate(), file)
        with open('diffuser.pkl', 'wb') as file:
            pickle.dump(self.diffuser(), file)

        # Re-create the binary decision space for use in the solver function
        binary_decision_space = []
        for x in self.search_space:
            s = format(x, "0" + str(self.outputs - 1) + "b")
            binary_decision_space.append(s)
        control_qubits = []
        x_gates = []
        minimum = np.random.choice(self.search_space)
        position = self.search_space.index(minimum)
        # Loop to set up initial values of the control qubits and where to apply x-gates
        for i in range(self.outputs - 1):
            if (binary_decision_space[position][i] == '0'):
                control_qubits.append(i + self.inputs)
                x_gates.append(i + self.inputs)
            if (binary_decision_space[position][i] == '1'):
                control_qubits.append(i + self.inputs)
                x_gates.append(i + self.inputs)
                break

        with open('control_qbits.pkl', 'wb') as file:
            pickle.dump(control_qubits, file)
        with open('xgates.pkl', 'wb') as file:
            pickle.dump(x_gates, file)

        # Create a list of input and output qubits for use in the solver function
        inputs = []
        for i in range(self.inputs):
            inputs.append(i)

        with open('inputs.pkl', 'wb') as file:
            pickle.dump(inputs, file)
        print("All circuits saved.")

    # Function to solve the quantum circuit
    def solve(self):

        # Load Objects
        with open('xgates.pkl', 'rb') as file:
            x_gates = pickle.load(file)
        with open('control_qbits.pkl', 'rb') as file:
            control_qubits = pickle.load(file)
        with open('state_prep.pkl', 'rb') as file:
            state_prep = pickle.load(file)
        with open('u_gate.pkl', 'rb') as file:
            u_gate = pickle.load(file)
        with open('diffuser.pkl', 'rb') as file:
            diffuser = pickle.load(file)
        with open('inputs.pkl', 'rb') as file:
            inputs = pickle.load(file)

        # Set up the variables to check for the minimum value
        minimum = np.random.choice(self.search_space)
        count = []
        current_minimums = []
        transpiled_grover_circuit = 0


        # Start preforming the grover operations
        for i in range(self.outputs - 1):

            register = QuantumRegister(self.inputs + self.outputs)
            oracle = QuantumCircuit(register)

            # Oracle for the Grover Circuit
            oracle.x(x_gates)
            oracle.mcx(control_qubits, self.inputs + self.outputs - 1)
            oracle.x(x_gates)

            # Compose the grover circuit
            cr = ClassicalRegister(self.inputs)
            step_1 = state_prep.compose(u_gate)
            step_1.barrier()
            step_2 = step_1.compose(oracle)
            step_2.barrier()
            step_3 = step_2.compose(u_gate)
            step_3.barrier()
            grover_circuit = step_3.compose(diffuser)
            grover_circuit.barrier()
            grover_circuit.add_register(cr)
            grover_circuit.swap(0, 2)
            grover_circuit.measure(inputs, inputs)

            # Run the circuit
            qasm_sim = Aer.get_backend('qasm_simulator')
            transpiled_grover_circuit = transpile(grover_circuit, qasm_sim)
            results = qasm_sim.run(transpiled_grover_circuit, shots=1).result()
            counts = results.get_counts()

            for measured_value in counts:
                a = int(measured_value[::1], 2)

            if ((self.search_space[a] < minimum) and (len(count) == 0)):
                minimum = self.search_space[a]
                control_qubits.append(len(control_qubits) + self.inputs)
                x_gates.append(len(x_gates) + self.inputs)

            elif (self.search_space[a] >= minimum):
                count.append(0)
                # if everything is 0
                if len(control_qubits) != self.outputs - 1:
                    control_qubits.append(len(control_qubits) + self.inputs)

                # Update when you need to mark 1 on the register
                if (len(count) == 1):
                    x_gates[-1] = x_gates[-1] + 1

                else:
                    if len(x_gates) != self.outputs - 1:
                        x_gates.append(x_gates[-1] + 1)

            current_minimums.append(self.search_space[a])

        with open('gover.pkl', 'wb') as file:
            pickle.dump(transpiled_grover_circuit, file)
        smallest_val = min(current_minimums)
        print("Smallest Delay: " + str(smallest_val) + " Path: " + str(a))


class ClassicalMinimizer:

    def __init__(self, search_space):
        self.search_space = search_space

    def solve(self):
        smallest_value = min(self.search_space)
        path = self.search_space.index(smallest_value)
        print("Smallest Delay: " + str(smallest_value) + " Path: " + str(path))


quantum_circuit = QuantumMinimizer(3, 11, decision_space)
classical_circuit = ClassicalMinimizer(decision_space)

qasm_sim = Aer.get_backend('qasm_simulator')
with open('gover.pkl', 'rb') as file:
    transpiled_grover_circuit = pickle.load(file)

#Run quantum_circuit.save_objects and quantum_circuit.solve before testing

def algorithm_solver():
    #qasm_sim.run(transpiled_grover_circuit, shots=1).result()
    classical_circuit.solve()

if __name__=='__main__':
    t = Timer("algorithm_solver()", "from __main__ import algorithm_solver")
    print(t.repeat(1, number=1))

