"""탐지니 Streamlit 진입점: 다단계 알약 탐지 및 식별 보조 MVP."""

from __future__ import annotations

import hashlib
import io

import streamlit as st

from tamgenie.detector import Detector
from tamgenie.pipeline import annotate, crop_image, overlap_flags, read_image, route
from tamgenie.question_engine import filter_candidates, load_catalog, next_question
from tamgenie.settings import get_settings

st.set_page_config(page_title="탐지니 | TamGenie", page_icon="💊", layout="wide")


@st.cache_resource(show_spinner="기존 YOLO11n 모델을 준비하고 있어요…")
def detector_resource(
    weights: str, modified: int, class_map: str, metadata: str, catalog_version: tuple
) -> Detector:
    """업로드별 모델 재로딩을 피하고 변경된 가중치·매핑은 다시 검증한다."""
    from pathlib import Path

    return Detector(Path(weights), load_catalog(Path(class_map), Path(metadata)))


def questions_panel(catalog: dict, primary_id: int, session_key: str) -> None:
    """관측 답변과 검증 속성을 비교하고 불확실성이 남으면 식별 불가로 끝낸다."""
    answer_key = f"answers:{session_key}"
    answers = st.session_state.setdefault(answer_key, {})
    candidates = filter_candidates(catalog, set(catalog), answers)
    question = next_question(catalog, candidates, set(answers))
    st.caption(
        "학습 JSON의 각인·색상을 정답으로 복사하지 않았습니다. "
        "정보가 없는 후보는 답변만으로 제외하지 않습니다."
    )
    if answers:
        st.write("입력한 관찰:", answers)
    if primary_id not in candidates:
        st.warning(
            "관찰한 특징이 1차 모델 후보의 검증 속성과 충돌합니다. "
            "모델을 정답으로 확정하지 않습니다."
        )
    if st.button("질문 답변 초기화", key=f"reset:{session_key}"):
        st.session_state[answer_key] = {}
        st.session_state.pop(f"ended:{session_key}", None)
        st.rerun()
    ended = st.session_state.get(f"ended:{session_key}", False)
    if question is not None and not ended:
        st.markdown("**3차 · 스무고개형 식별 보조**")
        st.caption(
            f"이번 질문에 출처가 있는 후보 {question.known_count}종 / "
            f"속성 미확보로 남겨 두는 후보 {question.unknown_count}종"
        )
        choice = st.selectbox(
            question.title,
            ["선택해주세요", *question.options, "모르겠어요"],
            key=f"question:{session_key}:{question.field}",
        )
        if st.button("관찰 적용", key=f"apply:{session_key}", disabled=choice == "선택해주세요"):
            st.session_state[answer_key] = {**answers, question.field: choice}
            st.rerun()
        if st.button("확인하기 어려워요 · 식별 불가로 종료", key=f"end:{session_key}"):
            st.session_state[f"ended:{session_key}"] = True
            st.rerun()
    else:
        if len(candidates) == 0:
            st.warning("식별 불가 · 입력한 관찰과 맞는 카탈로그 후보를 찾지 못했습니다.")
        elif len(candidates) == 1:
            st.warning(
                "식별 불가 · 조건과 맞는 참고 후보는 1종이지만 약품 정체를 확정할 수는 없습니다."
            )
        else:
            st.warning("식별 불가 · 후보를 구분할 검증 정보가 부족하거나 확인을 중단했습니다.")
    with st.expander(f"남은 참고 후보 {len(candidates)}종과 근거"):
        st.caption(
            "1차 모델 후보를 먼저 표시하고 나머지는 클래스 번호순입니다. "
            "표의 다른 후보에 대한 모델 점수는 생성하지 않았습니다."
        )
        rows = []
        for index in sorted(candidates, key=lambda index: (index != primary_id, index)):
            product = catalog[index]
            rows.append(
                {
                    "제품 코드": product.code,
                    "학습 라벨 이름": product.name,
                    "모델 신호": "1차 모델 후보" if index == primary_id else "점수 미제공",
                    "검증 속성": ", ".join(product.attributes) or "미확보 · 제외 근거 없음",
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")


def main() -> None:
    """사용자 이미지와 기존 모델을 연결하고 원본을 별도 데이터셋으로 저장하지 않는다."""
    settings = get_settings()
    st.title("탐지니 · TamGenie")
    st.subheader("다단계 AI 기반 알약 탐지 및 식별 보조 시스템")
    st.info(
        "사진에서 알약 위치와 참고 후보를 찾아봅니다. "
        "모델 점수는 실제 정답 확률이 아니며, 결과로 약품 정체나 복용 여부를 판단할 수 없습니다."
    )
    with st.sidebar:
        st.header("탐지 설정")
        detection_threshold = st.slider(
            "후보 표시 점수 (임시)", 0.05, 0.50, 0.25, 0.05, key="detection_threshold"
        )
        review_threshold = st.slider(
            "추가 확인 점수 기준 (임시)", 0.10, 0.95, 0.60, 0.05, key="review_threshold"
        )
        st.caption(
            "표시 기준을 낮추면 놓쳤던 알약을 추가로 찾을 수 있지만 오탐도 늘 수 있습니다. "
            "두 기준과 품질 기준은 검증 전 임시 설정입니다."
        )
        st.divider()
        st.write("1차 · YOLO11n")
        st.write("2차 · EfficientNet-B0 **미구현**")
        st.caption("학습된 2차 가중치가 없어 해당 단계는 건너뜁니다.")
        st.write("3차 · Interactive Identification")
    if not settings.class_map.is_file() or not settings.metadata.is_file():
        st.error("클래스 매핑 또는 출처 메타데이터 설정 파일이 없습니다.")
        st.stop()
    try:
        catalog = load_catalog(settings.class_map, settings.metadata)
    except (ValueError, OSError, KeyError) as error:
        st.error(f"카탈로그 설정을 확인해주세요: {error}")
        st.stop()
    known = sum(bool(product.attributes) for product in catalog.values())
    st.caption(
        f"현재 1차 모델 범위 {len(catalog)}종 · 출처 대조 속성이 있는 제품 {known}종. "
        "학습 범위 밖 약품이 비슷하게 예측될 수 있습니다."
    )
    with st.expander("질문에 사용하는 속성과 출처"):
        st.write(
            "현재 모양·색상 일부만 대조했습니다. "
            "각인·분할선 질문은 구분 가능한 검증 정보가 확보될 때만 생성합니다."
        )
        for product in catalog.values():
            for field, record in product.attributes.items():
                st.write(f"{product.name} · {field}: {', '.join(record['values'])}")
                st.link_button("공식 제품 설명", record["source_url"])
    uploaded = st.file_uploader(
        "알약 사진을 올려주세요 · 여러 알약이 함께 있어도 괜찮아요",
        type=["jpg", "jpeg", "png", "webp"],
        key="upload",
    )
    if not settings.weights.is_file():
        st.warning(
            "기존 best.pt가 연결되지 않았습니다. 실행 환경의 TAMGENIE_WEIGHTS를 설정해주세요."
        )
    if uploaded is None:
        st.write("사진 업로드 → 위치 탐지·Crop → 추가 확인 → 참고 후보 또는 식별 불가")
        st.caption(
            "반사·가림을 줄이고 각인과 알약 전체가 보이게 촬영해주세요. "
            "원본과 기존 가중치는 변경하지 않습니다."
        )
        return
    data = uploaded.getvalue()
    try:
        image = read_image(data, settings)
    except ValueError as error:
        st.error(str(error))
        return
    catalog_version = (settings.class_map.stat().st_mtime_ns, settings.metadata.stat().st_mtime_ns)
    weights_version = (
        str(settings.weights.resolve()),
        settings.weights.stat().st_mtime_ns if settings.weights.is_file() else None,
    )
    signature = (
        hashlib.sha256(data).hexdigest(),
        detection_threshold,
        review_threshold,
        catalog_version,
        weights_version,
    )
    if st.session_state.get("analysis", {}).get("signature") != signature:
        st.session_state.pop("analysis", None)
    if st.button(
        "분석하기", type="primary", key="analyze", disabled=not settings.weights.is_file()
    ):
        try:
            detector = detector_resource(
                str(settings.weights),
                settings.weights.stat().st_mtime_ns,
                str(settings.class_map),
                str(settings.metadata),
                catalog_version,
            )
            with st.spinner("YOLO11n으로 위치와 참고 후보를 확인하고 있어요…"):
                detections = detector.predict(image, detection_threshold)
            st.session_state["analysis"] = {"signature": signature, "detections": detections}
        except Exception as error:
            st.error(f"모델 추론을 완료하지 못했습니다: {error}")
            return
    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.image(image, caption="업로드 미리보기", width="stretch")
        return
    detections = analysis["detections"]
    if not detections:
        st.warning("식별 불가 · 탐지된 박스가 없습니다. 실제 알약이 없다는 뜻은 아닙니다.")
        st.image(image, width="stretch")
        return
    marked = annotate(image, detections, catalog, settings)
    st.image(
        marked,
        caption=f"탐지 박스 {len(detections)}개 · 실제 알약 개수를 확정한 값이 아닙니다.",
        width="stretch",
    )
    result_bytes = io.BytesIO()
    marked.save(result_bytes, format="PNG")
    st.download_button(
        "박스 표시 이미지 내려받기", result_bytes.getvalue(), "tamgenie_detection.png", "image/png"
    )
    conflicts = overlap_flags(detections)
    st.header("알약별 참고 결과")
    for detection in detections:
        product = catalog[detection.class_id]
        crop = crop_image(image, detection)
        outcome = route(detection, crop, review_threshold, flags=conflicts[detection.index])
        with st.container(border=True):
            left, right = st.columns([1, 2])
            with left:
                st.image(crop, caption=f"박스 #{detection.index} Crop", width="stretch")
                st.caption(f"원본 좌표 xyxy: {detection.bbox}")
            with right:
                st.subheader(f"#{detection.index} · {product.name}")
                st.write(f"제품 코드 {product.code} · 1차 모델 점수 **{detection.score:.3f}**")
                st.caption("실제 정답 확률이 아닌 모델 원시 점수입니다.")
                st.write(outcome["status"])
                if outcome["reasons"]:
                    st.warning(" / ".join(outcome["reasons"]))
                st.caption("2차 EfficientNet-B0: 미구현 · 학습 가중치 없음으로 건너뜀")
                scope = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
                key = f"{scope}:{detection.index}"
                if st.checkbox(
                    "질문으로 추가 확인", value=outcome["needs_review"], key=f"ask:{key}"
                ):
                    questions_panel(catalog, detection.class_id, key)
                else:
                    st.info(
                        "확정되지 않은 참고 후보입니다. "
                        "높은 점수도 정체의 정확성을 보장하지 않습니다."
                    )


if __name__ == "__main__":
    main()
