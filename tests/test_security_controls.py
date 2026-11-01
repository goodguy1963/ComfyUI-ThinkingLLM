import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from thinkingllm_core.audio_security import (
    AUDIO_ALLOWED_ROOTS_ENV,
    MAX_AUDIO_FILE_BYTES,
    MAX_AUDIO_SECONDS,
    resolve_local_audio_path,
    validate_audio_duration,
    validate_audio_file,
    validate_audio_shape,
)
from thinkingllm_core.hf_trust import (
    TRUSTED_HF_MODELS_ENV,
    get_trusted_hf_revision,
    mark_trusted_hf_snapshot,
    trust_remote_code_for_snapshot,
)


class TestAudioPathSecurity(unittest.TestCase):
    def test_audio_path_must_stay_under_an_allowed_root(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            inside_file = Path(allowed) / "inside.wav"
            inside_file.touch()
            outside_file = Path(outside) / "private.wav"
            outside_file.touch()

            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(inside_file.resolve(), resolve_local_audio_path(inside_file, default_roots=[allowed]))
                with self.assertRaisesRegex(ValueError, "inside an allowed audio directory"):
                    resolve_local_audio_path(outside_file, default_roots=[allowed])

    def test_audio_path_symlink_cannot_escape_allowed_root(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            outside_file = Path(outside) / "private.wav"
            outside_file.touch()
            link = Path(allowed) / "linked.wav"
            try:
                link.symlink_to(outside_file)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "inside an allowed audio directory"):
                    resolve_local_audio_path(link, default_roots=[allowed])

    def test_admin_audio_roots_and_processing_limits_are_enforced(self):
        with tempfile.TemporaryDirectory() as configured:
            audio = Path(configured) / "audio.wav"
            audio.touch()
            with mock.patch.dict(os.environ, {AUDIO_ALLOWED_ROOTS_ENV: configured}, clear=True):
                self.assertEqual(audio.resolve(), resolve_local_audio_path("audio.wav", default_roots=[]))
                with self.assertRaisesRegex(ValueError, f"{MAX_AUDIO_SECONDS}-second"):
                    validate_audio_duration((MAX_AUDIO_SECONDS + 1) * 16000, 16000)
                with self.assertRaisesRegex(ValueError, "sample rate"):
                    validate_audio_duration(16000, 384000)
                with self.assertRaisesRegex(ValueError, "sample limit"):
                    validate_audio_shape((1, 2, MAX_AUDIO_SECONDS * 96_000), 96000)

            with tempfile.TemporaryDirectory() as root:
                too_large = Path(root) / "large.wav"
                with too_large.open("wb") as stream:
                    stream.truncate(MAX_AUDIO_FILE_BYTES + 1)
                with self.assertRaisesRegex(ValueError, "size limit"):
                    validate_audio_file(too_large)


class TestHuggingFaceRemoteCodeTrust(unittest.TestCase):
    def test_trust_requires_explicit_repo_commit_and_matching_snapshot_marker(self):
        repo_id = "example/custom-model"
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / revision
            snapshot.mkdir()
            with mock.patch.dict(os.environ, {TRUSTED_HF_MODELS_ENV: f"{repo_id}@{revision}"}, clear=True):
                self.assertEqual(revision, get_trusted_hf_revision(repo_id))
                self.assertFalse(trust_remote_code_for_snapshot(snapshot, repo_id))
                mark_trusted_hf_snapshot(snapshot, repo_id, revision)
                self.assertTrue(trust_remote_code_for_snapshot(snapshot, repo_id))
                self.assertFalse(trust_remote_code_for_snapshot(snapshot, "example/other-model"))

                os.environ["THINKINGLLM_COMMERCIAL_RELEASE"] = "1"
                self.assertFalse(trust_remote_code_for_snapshot(snapshot, repo_id))

    def test_trust_configuration_rejects_non_commit_revisions(self):
        with mock.patch.dict(os.environ, {TRUSTED_HF_MODELS_ENV: "example/model@main"}, clear=True):
            with self.assertRaisesRegex(ValueError, "40-character commit SHA"):
                get_trusted_hf_revision("example/model")


if __name__ == "__main__":
    unittest.main()
