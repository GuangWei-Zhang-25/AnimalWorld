# Pig: a second species in two YAML files

`animalworld init pig_demo --species pig` copies `project.yaml` and `slots.yaml`; there are no bundled abstracts, so the first stage searches Europe PMC (network needed; or set `literature.backend: offline` and `literature.source` to your own CSV / JSONL of abstracts).
`animalworld pipeline pig_demo --auto-accept --train-dry-run` then runs every stage with the mock generator and the mock proposer; training is only printed, not executed (40 accepted images are below the 200-image threshold).
The real backends are set exactly as in `examples/mouse/README.md`.
