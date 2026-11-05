# pillbbox

알약 이미지에서 알약의 위치(bbox)와 종류를 검출하는 프로젝트입니다.

- 입력: 알약 촬영 이미지 / 출력: 알약별 bbox와 클래스
- 모델: 미정. 여러 모델을 실험해 본 뒤 확정합니다.
- 실험 기록: [wandb](#실험-기록-wandb)

## 개인 제출용 YOLO11n 베이스라인

**내부 데이터만 사용하는 YOLO11n 사전학습 베이스라인의 20epoch 학습·평가를 완료했습니다.**
설정은 imgsz640 / batch8 / seed42이며, epoch별 train+val 누적 시간은98.51초입니다.
개발용 validation에서 mAP50 **0.7375**, mAP50-95 **0.7104**,
Precision **0.8177**, Recall **0.6402**를 기록했습니다.
검증에 포함된29클래스에 대한 결과이며 독립 test 성능이 아닙니다.
학습 코드는 기본적으로 설정 확인만 하며 새 학습은 `--confirm-training`을 지정해야 시작됩니다.

현재 결과는 PASS197장/REVIEW35장, 라벨660개/56클래스입니다.
촬영·유사 그룹을 그대로 보존한 최신 분할은 train158장/val39장(약80.2:19.8)입니다.
학습56클래스·검증29클래스는 유지했으며 검증에 없는27클래스의 성능은 평가할 수 없습니다.
원본232장·JSON763개의 해시 보존과 전체660개 라벨의 좌표 역변환 검증을 통과했습니다.
PASS는 구조 검사 통과 후보이며 약품 정체의 정확성 보증이 아닙니다.

- [환경·데이터 구조 검사 결과](docs/YOLO11N_PREFLIGHT.md)
- [최종 보고서: 실제 학습·평가 결과와 한계](docs/FINAL_REPORT.md)
- [제출용 PDF: 베이스라인과 Recall 분석 요약](output/pdf/pill_ai_final_report.pdf)
- [제출 자료 확인과 PDF 재생성 방법](docs/SUBMISSION_GUIDE.md)
- [클래스별 평가 지표](docs/results/yolo11n_20261105/per_class_metrics.csv)
- [Confidence 5종 비교와 저성능 클래스 원인 분석](docs/RECALL_ANALYSIS.md)
- [라벨 QA 설계 제안](docs/DATA_QA_DESIGN.md)
- [내부 EDA](notebooks/01_internal_data_eda.ipynb), [외부 EDA](notebooks/02_external_data_eda.ipynb)
- 검증·변환: `data_process/prepare_internal_baseline.py`
- 학습: `models/train.py`, 평가: `models/evaluate.py`, 보고서: `models/export_baseline_report.py`

**RTX 5070 환경 주의:** 이 저장소의 기본 Python 3.12/PyTorch cu126 환경은 GPU를 인식해도
실제 연산이 실패했습니다. 현재 PC에서 확인한 정상 조합은 Python 3.14.2,
PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, Ultralytics 8.4.193입니다.
아래의 `--no-project --python 3.14`는 이 검증된 설치를 명시적으로 사용합니다.
`uv sync`만으로 같은 GPU 환경이 재현된다고 주장하지 않습니다.
기존 PyTorch/CUDA 설치 환경은 변경하지 않았습니다.
그룹 유지·사진 비율 우선의 정확한 분할 계산을 위해 기존 설치된 SciPy를 직접 의존성으로 명시했습니다.

```powershell
# 우선 환경변수에 실제 로컬 데이터·EDA 결과·새 출력 폴더를 지정한다.
# 이 변환 명령은 전체 내부 QA·준비를 승인받은 뒤 실행한다.
uv run --no-project --python 3.14 python -m data_process.prepare_internal_baseline --data-dir $env:PILL_INTERNAL_DATA_DIR --eda-dir $env:PILL_INTERNAL_EDA_DIR --output-dir $env:PILL_YOLO_DATA_DIR --review-csv configs/internal_review_exclusions_20261105.csv

# 새로 생성한 데이터셋의 복사 해시·좌표·그룹 누수까지 대조한다.
uv run --no-project --python 3.14 python -m data_process.verify_internal_dataset --dataset $env:PILL_YOLO_DATA_DIR --raw-root $env:PILL_INTERNAL_DATA_DIR

# QA·그룹·사진·라벨을 그대로 두고 80:20 비율을 우선해 새 폴더에 재분할한다.
uv run --no-project --python 3.14 python -m data_process.resplit_internal_baseline --source $env:PILL_YOLO_PREVIOUS_DIR --output $env:PILL_YOLO_DATA_DIR

# 기본은 설정 확인만 하며 가중치 다운로드·학습을 하지 않는다.
uv run --no-project --python 3.14 python -m models.train --data "$env:PILL_YOLO_DATA_DIR/data.yaml" --output-dir $env:PILL_RUN_DIR

# 데이터 준비 결과를 검토하고 학습까지 별도 승인한 뒤에만 --confirm-training을 추가한다.
uv run --no-project --python 3.14 python -m models.train --data "$env:PILL_YOLO_DATA_DIR/data.yaml" --output-dir $env:PILL_RUN_DIR --confirm-training

# 학습 완료 뒤 validation 지표·confusion matrix·예측 예시를 별도 저장한다.
uv run --no-project --python 3.14 python -m models.evaluate --weights "$env:PILL_RUN_DIR/best.pt" --data "$env:PILL_YOLO_DATA_DIR/data.yaml" --output-dir $env:PILL_EVAL_DIR
```

실제 준비 시 기존 EDA의 보류 CSV도 `--legacy-quarantine`, `--legacy-review-log`,
반복 가능한 `--review-csv`로 명시할 수 있습니다. 현재 보류35장을 재현하는 메타데이터는
`configs/internal_review_exclusions_20261105.csv`에 있습니다. 이 표는 원본 이미지·JSON을 포함하지 않습니다.
재생성할 때는 기존 결과를 덮어쓰지 않도록 새로운 출력 폴더를 사용하세요.
최신 준비 폴더는 `internal_yolo11n_8020_20261105`이며 기존65:35 폴더도 보존했습니다.
GitHub용 EDA 사본은 이미지 첨부와 코드 출력을 제거했습니다.
로컬 비교 그림은 `PILL_INTERNAL_EDA_DIR` / `PILL_EXTERNAL_EDA_DIR`로 다시 연결할 수 있습니다.
원본 데이터·가중치·실행 출력은 커밋하지 않습니다.

<!-- 채워 주세요: 목표 지표(예: test mAP50-95), 데이터 출처, 일정, 팀원과 역할 -->

## 문서

| 문서 | 내용 |
|---|---|
| [docs/CODE_CONVENTION.md](docs/CODE_CONVENTION.md) | 코드 컨벤션 (포맷, 이름, import, 경로·장치) |
| [docs/GIT_CONVENTION.md](docs/GIT_CONVENTION.md) | Git 컨벤션 (커밋 메시지, 브랜치, PR) |
| [data/README.md](data/README.md) | annotation JSON 구조 |
| [AGENTS.md](AGENTS.md) | AI 코딩 에이전트(Codex, Claude Code) 지침 |

## 치트시트

자주 쓰는 명령만 모았습니다. 자세한 설명은 아래 각 섹션에 있습니다.

| 상황 | 명령 |
|---|---|
| 처음 받았을 때 (한 번) | `uv sync` → `git config core.hooksPath .githooks` |
| pull 받은 뒤 | `uv sync` |
| 코드 실행 | `uv run python main.py` |
| 노트북 열기 | `uv run jupyter lab` |
| 패키지 추가 / 제거 | `uv add <패키지>` / `uv remove <패키지>` |
| 개발용 패키지 추가 | `uv add --dev <패키지>` |
| 커밋 전 코드 정리 | `uv run ruff format .` → `uv run ruff check . --fix` |
| Mac에서 MPS 에러 | `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python main.py` |

작업 흐름:

```bash
git switch main && git pull && uv sync      # 최신 상태로 맞추기
git switch -c feat/<짧은-설명>                # 브랜치 만들기
# ... 작업 ...
uv run ruff format . && uv run ruff check . --fix
git add <파일> && git commit -m "feat: <요약>"
git push -u origin feat/<짧은-설명>
gh pr create --base main                     # PR 만들기 (아래 "PR 올리기" 참고)
```

- 패키지를 바꿨다면 `pyproject.toml`과 `uv.lock`을 함께 커밋합니다.
- 노트북은 출력을 지운 뒤 커밋합니다.
- `pip install`은 쓰지 않습니다.

## 개발 환경

이 프로젝트는 [uv](https://docs.astral.sh/uv/)로 Python 버전과 패키지를 관리합니다.

- Python 3.12 (`.python-version`에 고정)
- 패키지 목록: `pyproject.toml`
- 정확한 설치 버전: `uv.lock`

### 1. uv 설치

```bash
# macOS
brew install uv

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 환경 만들기

```bash
uv sync
```

`.venv`가 만들어지고 `uv.lock`에 적힌 버전 그대로 설치됩니다. Python 3.12가 없으면 uv가 알아서 받아 옵니다.

같이 git hook도 켭니다. `main`에 직접 커밋·push하는 실수를 막아 줍니다.

```bash
git config core.hooksPath .githooks
```

OS마다 PyTorch가 다르게 설치됩니다.

| OS | PyTorch | 사용 장치 |
|---|---|---|
| macOS | PyPI 기본 빌드 | `mps` (Apple GPU) |
| Windows / Linux | CUDA 12.6 빌드 | `cuda` (NVIDIA GPU) |

NVIDIA GPU를 쓰려면 CUDA 12.6을 지원하는 드라이버가 설치되어 있어야 합니다.

### 3. 실행

```bash
uv run python main.py
uv run jupyter lab
```

`uv run`을 붙이면 가상환경을 따로 활성화하지 않아도 됩니다. VS Code에서는 인터프리터 / 커널로 `.venv`를 선택하세요.

## 패키지 관리

```bash
uv add <패키지>          # 추가
uv add "<패키지>>=1.2"   # 버전 범위 지정
uv add --dev <패키지>    # 개발용 (jupyter, linter 등)
uv remove <패키지>       # 제거
```

- 패키지를 추가하거나 제거했다면 **`pyproject.toml`과 `uv.lock`을 함께 커밋**하세요.
- pull 받은 뒤 다른 사람이 패키지를 바꿨다면 `uv sync`를 다시 실행하세요.
- `pip install`은 쓰지 마세요. 설치는 되지만 `pyproject.toml`에 기록되지 않아 다른 팀원 환경에 반영되지 않습니다.

## 장치 선택

`src/config.py`의 `get_device()`가 실행 환경에 맞는 장치를 골라 줍니다 (cuda → mps → cpu 순).

```python
from src.config import get_device

model.train(..., device=get_device())
```

macOS에서 MPS가 지원하지 않는 연산 때문에 에러가 나면, 그 연산만 CPU로 돌리도록 실행하세요.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python main.py
```

## 실험 기록 (wandb)

처음 한 번, `.env.example`을 `.env`로 복사하고 [API 키](https://wandb.ai/authorize)를 채웁니다. `.env`는 커밋되지 않습니다.

```bash
cp .env.example .env   # WANDB_API_KEY, WANDB_ENTITY(팀 이름) 입력
```

학습 전에 모델을 넘겨 `init_wandb()`를 호출하면 그 모델의 학습 로그(loss, mAP, 그래프, 가중치)가 wandb에 기록됩니다. ultralytics 전역 설정은 건드리지 않으므로 이 프로젝트 밖의 YOLO 학습에는 영향이 없습니다.

```python
from ultralytics import YOLO

from src.config import get_device, init_wandb

model = YOLO("yolov8s.pt")
init_wandb(model, run_name="yolov8s-baseline", config={"seed": 42})
model.train(data="data.yaml", epochs=50, device=get_device())
```

| 상황 | 명령 |
|---|---|
| wandb 끄고 실행 | `WANDB_MODE=disabled uv run python main.py` |
| 인터넷 없이 기록만 남기기 | `WANDB_MODE=offline uv run python main.py` |
| offline 기록 나중에 올리기 | `uv run wandb sync wandb/offline-run-*` |

## 코드 컨벤션

[docs/CODE_CONVENTION.md](docs/CODE_CONVENTION.md)를 참고하세요. 커밋 전에 아래 두 명령을 실행합니다.

```bash
uv run ruff format .
uv run ruff check . --fix
```

VS Code에서는 추천 확장(Ruff)을 설치하면 저장할 때 자동으로 정리됩니다. ruff 규칙은 `pyproject.toml`에만 둡니다.

## Git 컨벤션

자세한 내용은 [docs/GIT_CONVENTION.md](docs/GIT_CONVENTION.md)를 참고하세요. 핵심만 요약하면:

- `main`에 직접 커밋·push하지 않고, `<type>/<짧은-설명>` 브랜치를 만들어 PR로 합칩니다.
- 커밋 메시지는 `<type>: <요약>` (한국어, 50자 이내, 마침표 없음). type은 `feat` `fix` `exp` `data` `refactor` `docs` `chore`.
- PR 본문은 [PR 템플릿](.github/PULL_REQUEST_TEMPLATE.md)을 채웁니다.

## AI 코딩 에이전트

Codex, Claude Code 등 에이전트용 지침은 [AGENTS.md](AGENTS.md) 하나로 관리합니다.

- Codex는 `AGENTS.md`를 바로 읽습니다.
- Claude Code는 `CLAUDE.md`를 읽고, `CLAUDE.md`가 `AGENTS.md`를 불러옵니다.
- 규칙을 바꿀 때는 `AGENTS.md`만 고칩니다. 개인 설정은 `CLAUDE.local.md` 등 커밋되지 않는 파일에 둡니다.
- 에이전트가 만든 코드도 커밋 전에 직접 읽고, 위 컨벤션과 PR 템플릿 체크리스트를 똑같이 적용합니다.

## 폴더 구조

```
pillbbox/
├── data/               # 데이터 (구조: data/README.md, 이미지·annotation은 커밋 안 함)
├── docs/               # 컨벤션 등 문서
├── data_process/       # 전처리, 라벨 생성, 데이터셋 yaml 생성
├── models/             # 학습, 평가, 튜닝
├── notebooks/          # 실험 노트북
├── team_test/          # 팀원 Git 연습용
├── src/                # 공통 설정 (config.py)
├── AGENTS.md           # AI 에이전트 지침 (CLAUDE.md는 이 파일을 불러옴)
├── CLAUDE.md
├── main.py
├── visualization.py
├── pyproject.toml
└── uv.lock
```
