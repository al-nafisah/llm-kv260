# Build logs

One file per measurement, numbered in order: `0001-short-title.md`. Raw output
goes in `evidence/` under the same number, as `.txt`, so every number in a log
traces back to the text it was read from.

Each log has the same shape:

- **Status** on the first line: `Attested`, with the date and the boards and
  builds it came from.
- **Result**: the numbers, each pointing at its evidence file.
- **What this does and does not establish**: the claim and its limits.
- **Configuration**: tool versions and settings, enough to repeat the build.
- **Traps**: whatever cost time, so it does not cost it twice.

See [../measurement.md](../measurement.md) for why the split between Attested
and Cited numbers is kept.
