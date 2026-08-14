from rmsynth_reference import (
    LinearMap,
    LinearPhaseProgram,
    PhasePolynomial,
    optimize,
    synthesize_program,
)

program = LinearPhaseProgram(PhasePolynomial(4, (1,) * 15), LinearMap.identity(4))
circuit = synthesize_program(program)
result = optimize(circuit)

print(f"T-count: {result.report.before_t_count} -> {result.report.after_t_count}")
