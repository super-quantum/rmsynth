from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from ._version import __version__
from .errors import ResourceLimitError, RMSynthError, VerificationError
from .io import circuit_to_data, read_circuit, write_json
from .optimizer import optimize
from .semantics import extract_program, program_digest
from .verify import verify_circuits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rmsynth-ref")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="summarize a circuit")
    inspect_parser.add_argument("input")

    optimize_parser = commands.add_parser("optimize", help="optimize a circuit")
    optimize_parser.add_argument("input")
    optimize_parser.add_argument("--output", type=Path, required=True)
    optimize_parser.add_argument("--report", type=Path, required=True)
    optimize_parser.add_argument("--force", action="store_true")

    verify_parser = commands.add_parser("verify", help="compare two circuits")
    verify_parser.add_argument("expected")
    verify_parser.add_argument("actual")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        handlers = {
            "inspect": _inspect,
            "optimize": _optimize,
            "verify": _verify,
        }
        return handlers[arguments.command](arguments)
    except ResourceLimitError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    except VerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 4
    except (RMSynthError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _inspect(arguments: argparse.Namespace) -> int:
    circuit = read_circuit(arguments.input)
    program = extract_program(circuit)
    print(
        json.dumps(
            {
                "operations": len(circuit.operations),
                "qubits": circuit.qubits,
                "semantic_digest": program_digest(program),
                "t_count": circuit.t_count,
            },
            sort_keys=True,
        )
    )
    return 0


def _optimize(arguments: argparse.Namespace) -> int:
    input_path = None if arguments.input == "-" else Path(arguments.input).resolve()
    output_path = arguments.output.resolve()
    report_path = arguments.report.resolve()
    if output_path == report_path:
        raise RMSynthError("output and report paths must differ")
    if input_path == report_path:
        raise RMSynthError("report path must differ from the input path")
    if not arguments.force:
        for path in (arguments.output, arguments.report):
            if path.exists():
                raise RMSynthError(f"output already exists: {path}")
    result = optimize(read_circuit(arguments.input))
    write_json(arguments.output, circuit_to_data(result.circuit), force=arguments.force)
    write_json(arguments.report, result.report.to_dict(), force=arguments.force)
    return 0


def _verify(arguments: argparse.Namespace) -> int:
    expected = read_circuit(arguments.expected)
    actual = read_circuit(arguments.actual)
    verification = verify_circuits(expected, actual)
    payload: dict[str, object] = {"equivalent": verification.equivalent}
    if verification.witness is not None:
        payload["witness"] = asdict(verification.witness)
    print(json.dumps(payload, sort_keys=True))
    return 0 if verification.equivalent else 4


if __name__ == "__main__":
    raise SystemExit(main())
