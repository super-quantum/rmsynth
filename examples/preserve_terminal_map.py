from rmsynth_reference import CNOT, Circuit, extract_program, optimize

circuit = Circuit(2, (CNOT(0, 1),))
result = optimize(circuit)

assert extract_program(result.circuit).linear_map.rows == (1, 3)
print(result.report.status)
