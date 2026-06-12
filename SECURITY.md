# Security & Data Notes

- **No data ships here.** Everything is synthetic, generated at runtime. There is
  no real or PHI-bearing data in the repo or the tests.
- **The benchmark is controlled, not real-world.** See `REPORT.md` → "What 96.5%
  does and doesn't mean" before quoting any accuracy number.
- **The normalizer never invents codes** — it chooses only from retrieved
  candidates. The optional LLM path is constrained the same way.

Report issues via GitHub issues; do not include real clinical data.
