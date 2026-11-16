# Security and reproducibility

This repository publishes a local research workflow. It does not require API keys
or credentials for inference. Obtain competition data through your own authorized
Kaggle account; never commit credentials, `.env` files, or `kaggle.json`.

## Recorded environment

`requirements.txt` is the complete package snapshot used for the submitted result,
not a minimal or security-updated installation profile. It is retained unchanged
so the competition result remains reproducible. It includes both OpenCV packages;
do not treat this historical snapshot as a recommended production dependency set.

A review on **16 November 2026** identified advisories affecting these pinned packages:

| Package | Recorded version | Fix version identified in the reviewed advisories |
| --- | --- | --- |
| certifi | 2026.7.22 | 2026.10.18 |
| filelock | 3.32.3 | 4.2.1 |
| Pillow | 12.3.0 | 13.0.0 |
| setuptools | 84.0.0 | 84.1.0 |
| urllib3 | 2.7.0 | 2.8.0 |

The issues include certificate trust, HTTPS proxy handling, malicious-image
decoding, local lock-file interference, and a legacy Windows installation path.
Exploitability depends on the platform, input files, permissions, and code path.
These fixes have **not** been validated here as a replacement environment, and
exact output parity with upgraded dependencies is not claimed. Consult the current
upstream advisories before selecting versions for new work:

- [certifi](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2)
- [filelock](https://github.com/tox-dev/filelock/security/advisories/GHSA-69hj-f68j-2879)
- [Pillow](https://github.com/python-pillow/Pillow/security/advisories/GHSA-5x8w-8jj5-q52h)
- [setuptools](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w)
- [urllib3](https://github.com/urllib3/urllib3/security/advisories/GHSA-8988-9cw3-xx77)

For historical reproduction, use an isolated environment and the official image
files. For new applications, test an updated dependency set and validate its
predictions before deployment. Do not expose the historical environment as a
service accepting arbitrary uploads.

## Model files

The `.pt` checkpoints contain serialized Python objects. Verify their hashes using
`inference.restore_weights()` and load only trusted, expected copies. A matching
hash establishes file identity; it does not make arbitrary model files safe.
The notebook checks both files before inference.

## Reporting

Report suspected credentials or exploitable flaws privately to the repository
owner through GitHub private vulnerability reporting if it is available. Do not
post secret values in public issues. Security review does not guarantee the
absence of every vulnerability.
