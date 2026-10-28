import ctypes
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import AILab_LlamaCppInstaller as installer


class LlamaCppInstallerTests(unittest.TestCase):
    def test_gpu_probe_registers_dynamic_backends_before_checking_offload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / "llama_cpp"
            backend_dir = package_dir / "lib"
            backend_dir.mkdir(parents=True)
            package_file = package_dir / "__init__.py"
            package_file.touch()

            supports_gpu = mock.Mock(return_value=True)
            low_level = SimpleNamespace(_lib=None, llama_supports_gpu_offload=supports_gpu)
            load_backends = mock.Mock()
            ggml = SimpleNamespace(ggml_backend_load_all_from_path=load_backends)
            modules = {
                "llama_cpp.llama_cpp": low_level,
                "llama_cpp": SimpleNamespace(__file__=str(package_file)),
                "llama_cpp._ggml": ggml,
            }

            with mock.patch.object(installer.importlib, "import_module", side_effect=modules.__getitem__):
                self.assertTrue(installer._probe_gpu_offload_support())

            load_backends.assert_called_once()
            self.assertIsInstance(load_backends.call_args.args[0], ctypes.c_char_p)
            self.assertEqual(load_backends.call_args.args[0].value, os.fsencode(backend_dir))
            supports_gpu.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
