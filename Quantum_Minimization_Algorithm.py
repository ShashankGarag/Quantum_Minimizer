import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import algorithm_globals
from qiskit.algorithms.amplitude_amplifiers.grover import Grover
from qiskit.algorithms import AmplificationProblem
from qiskit.circuit.library import XGate, ZGate
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.visualization import plot_histogram
from qiskit import transpile

decision_space = [5, 4, 12, 10, 8]


class QuantumMinimizer:

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.best_value = '0'
        self.control_qubits = [1]
        self.x_gates = [1]

    def state_preparation(self):
        # Create Circuit
        register = QuantumRegister(self.inputs + self.outputs)
        hadamards = QuantumCircuit(register)
        hadamards.h(list(range(self.inputs)))
        hadamards.x(self.inputs + self.outputs - 1)
        hadamards.h(self.inputs + self.outputs - 1)
        return hadamards

    def u_gate(self):
        register = QuantumRegister(self.inputs + self.outputs)
        initialize = QuantumCircuit(register)
        custom_gate = XGate().control(self.inputs)

        # Create a list of binary values from the decision space
        binary_decision_space = []
        for x in decision_space:
            s = format(x, "0" + str(self.outputs - 1) + "b")
            binary_decision_space.append(s)

        # Create list of binary indexes for each list value
        binary_indexes = []
        regular_indexes = []
        length = range(len(binary_decision_space))
        for index in length:
            bin_index = format(index, "0" + str(self.inputs) + "b")
            binary_indexes.append(bin_index)
        for i in length:
            regular_indexes.append(i)

        # Number of contol qubits
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
        return initialize

    def oracle(self):
        oracle_register = []
        oracle_register_new = []
        oracle_register_final = []
        oracle_register.extend(self.control_qubits)
        oracle_register.append(self.inputs + self.outputs)
        for i in oracle_register:
            oracle_register_new.append(i - 1)
        register = QuantumRegister(self.inputs + self.outputs)
        oracle = QuantumCircuit(register)
        custom_gate = XGate().control(self.control_qubits[-1])
        for x in range(len(str(self.x_gates[-1]))):
            if str(self.x_gates[-1])[x] == '1':
                oracle.x(register[x + self.inputs])
        for x in oracle_register_new[:-1]:
            x += self.inputs
            oracle_register_final.append(x)
        oracle_register_final.append(self.inputs + self.outputs - 1)
        oracle.append(custom_gate, oracle_register_final)
        for x in range(len(str(self.x_gates[-1]))):
            if str(self.x_gates[-1])[x] == '1':
                oracle.x(register[x + self.inputs])
        return oracle

    def diffuser(self):
        inputs = []
        for i in range(self.inputs):
            inputs.append(i)
        register = QuantumRegister(self.inputs + self.outputs)
        diffuser = QuantumCircuit(register)
        diffuser.h(range(self.inputs))
        diffuser.z(range(self.inputs))
        custom_gate = ZGate().control(self.inputs - 1)
        diffuser.append(custom_gate, inputs)
        diffuser.h(range(self.inputs))
        return diffuser

    def solve(self):

        "Not implemented yet:"
        """value = self.outputs - 1
        for digit in range(value):
            for segment in range(digit + 1):
                if result == self.best_value:  # Replace value[segment] with the value of the qubits there
                    self.x_gates.append(segment)
                    self.best_value += '0'
                else:
                    self.best_value += '0'
                self.control_qubits.append(self.control_qubits[-1] + 1)"""

        inputs = []
        outputs = []
        for i in range(self.inputs):
            inputs.append(i)
        for i in range(self.outputs):
            outputs.append(i)

        cr = ClassicalRegister(self.inputs)
        step_1 = self.state_preparation().compose(self.u_gate())
        step_1.barrier()
        step_2 = step_1.compose(self.oracle())
        step_2.barrier()
        step_3 = step_2.compose(self.u_gate().inverse())
        step_3.barrier()
        grover_circuit = step_3.compose(self.diffuser())

        grover_circuit.add_register(cr)
        grover_circuit.measure(inputs, inputs)

        qasm_sim = Aer.get_backend('qasm_simulator')
        transpiled_grover_circuit = transpile(grover_circuit, qasm_sim)
        results = qasm_sim.run(transpiled_grover_circuit, shots=100).result()
        counts = results.get_counts()

        return plot_histogram(counts)


circuit = QuantumMinimizer(3, 5)
circuit.solve()