"""실제 모델과 AppTest로 업로드 후 화면 전환을 재현하는 선택적 통합 검사다."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT


def run() -> None:
    """이미지는 환경변수로 지정하고 원본을 테스트에 복사하거나 커밋하지 않는다."""
    image_path = Path(os.environ["TAMGENIE_TEST_IMAGE"])
    os.environ["TAMGENIE_TEST_IMAGE"] = str(image_path.resolve())
    script = """
import os, runpy
from pathlib import Path
from unittest.mock import patch
class Upload:
    def getvalue(self):
        return Path(os.environ['TAMGENIE_TEST_IMAGE']).read_bytes()
with patch('streamlit.file_uploader', return_value=Upload()):
    runpy.run_path('app.py', run_name='__main__')
"""
    page = AppTest.from_string(script, default_timeout=90).run()
    assert not page.exception
    page.button(key="analyze").click().run()
    assert not page.exception, str(page.exception)
    detections = page.session_state["analysis"]["detections"]
    assert len(detections) >= 2, "여러 알약이 있는 실제 검증 이미지를 지정하세요."
    question_count = len(page.selectbox)
    if question_count:
        page.selectbox[0].select("모르겠어요").run()
        apply = next(button for button in page.button if button.key.startswith("apply:"))
        apply.click().run()
        assert not page.exception
    with TemporaryDirectory() as directory:
        invalid = Path(directory) / "invalid.png"
        invalid.write_bytes(b"invalid image data")
        os.environ["TAMGENIE_TEST_IMAGE"] = str(invalid)
        broken = AppTest.from_string(script, default_timeout=90).run()
        assert not broken.exception and len(broken.error) == 1
        blank = Path(directory) / "blank.png"
        Image.new("RGB", (640, 640), "black").save(blank)
        os.environ["TAMGENIE_TEST_IMAGE"] = str(blank)
        page.run()
        assert not page.exception
        try:
            page.session_state["analysis"]
        except KeyError:
            pass
        else:
            raise AssertionError("새 업로드에 이전 분석 결과가 남았습니다.")
    os.environ["TAMGENIE_TEST_IMAGE"] = str(image_path)
    result = {
        "actual_model_inference": True,
        "detections": len(detections),
        "question_panels": question_count,
        "unknown_answer_applied": bool(question_count),
        "invalid_upload_rejected": True,
        "new_upload_clears_previous_result": True,
        "ui_exceptions": 0,
        "trained": False,
    }
    output = PROJECT_ROOT / "outputs/tamgenie_tests"
    output.mkdir(parents=True, exist_ok=True)
    (output / "streamlit_smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
