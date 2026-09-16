# QEdark silicon form factor

`Si_f2.txt` is the tabulated silicon crystal form factor distributed with
[QEdark](https://github.com/adrian-soto/QEdark_repo) and used by the QEdark
Python analysis notebooks associated with arXiv:1509.01598.

- Grid: 900 momentum bins x 500 energy bins
- Momentum step: 0.02 alpha m_e
- Energy step: 0.1 eV
- SHA-256: `b136379e009d94963be0e64f7f14a6f5d5a0839d1bc90e91e6710e5b35b013da`

`DM_Rate_Integration_dev/migdal_Si.py` loads this repository-relative file by
default, so no external QEdark checkout or machine-specific path is required.
