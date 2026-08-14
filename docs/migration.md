# Migration from 0.0.1

Version 0.1 intentionally replaces the experimental 0.0.1 interface.

| 0.0.1 | 0.1 |
|---|---|
| Distribution `rmsynth` | Distribution `rmsynth-reference` |
| Import `rmsynth` | Import `rmsynth_reference` |
| Command `rmsynth-optimize` | Command `rmsynth-ref` |
| Multiple native and heuristic backends | One bounded C++ exact decoder |
| Unbounded or backend-specific options | Fixed limits and one algorithm |

There is no compatibility import or automatic conversion layer. Existing integrations should pin
0.0.1 until they have moved to the versioned circuit JSON or the new Python value objects.
