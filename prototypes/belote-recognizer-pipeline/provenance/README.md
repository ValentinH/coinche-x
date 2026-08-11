# Asset provenance

Only these sources feed this slice:

- Training: Andrew Tidey, `Cards Pack`, CC0 1.0. Page and immutable archive
  hash in `sources.json`.
- Holdout only: Brian Huisman / GreyWyvern, public domain with BSD-3-Clause
  fallback. Different artwork. Page and immutable archive hash in
  `sources.json`.

`npm run assets:acquire` downloads each archive, rejects a SHA-256 mismatch,
extracts only Belote's 32 cards, and writes per-file hashes to
`assets.sha256`. Original license texts are vendored here.

## Data firewall

The generator has no arbitrary source option. `train` and `validation` resolve
only to `training/sources/andrew-tidey`; `holdout` resolves only to
`holdout/sources/greywyvern`. The verifier rejects a crossed role/path.

Project smoke photos and their truth manifest are intentionally outside this
package and were not read, listed, copied, tuned against, or used for
templates.
