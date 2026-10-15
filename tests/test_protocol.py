import hashlib,json
import pytest
from otg.protocol import verify_final_protocol


def test_final_protocol_requires_freeze_and_detects_changed_artifacts(tmp_path):
    with pytest.raises(RuntimeError):verify_final_protocol(tmp_path)
    (tmp_path/'docs').mkdir();artifact=tmp_path/'model.bin';artifact.write_bytes(b'frozen')
    manifest={'status':'frozen','frozen_files':[{'path':'model.bin','sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
    (tmp_path/'docs/FINAL_PROTOCOL.json').write_text(json.dumps(manifest))
    protocol,digest=verify_final_protocol(tmp_path);assert protocol==manifest and len(digest)==64
    artifact.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='changed'):verify_final_protocol(tmp_path)
