# Test runtimes

| Command | Environment | Successful runtime | Measured | Recommended timeout |
| --- | --- | ---: | --- | ---: |
| `python -c "import sys,unittest; sys.path[:0]=['.','tests']; import test_new_features as m; unittest.main(m, argv=['unittest','-v'], exit=False)"` | Windows, ComfyUI embedded Python 3.12, CPU test path | 48.8 s | 2026-08-24 | 120 s |
