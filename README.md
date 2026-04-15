# Technical_portfolio_UDAGCN
서울과학기술대학교_데이터사이언스학과_24장수진_졸업심사_테크니컬포트폴리오
# 다중 도메인 지식 그래프 기반 링크 예측 및 관계 분류

이 프로젝트는 **AI, 분자생물학(MB), 항공우주공학(AE)** 으로 구성된 지식 그래프를 기반으로, 한 도메인에서 학습한 관계 패턴을 다른 도메인으로 전이하여 **링크 존재 여부와 관계 타입을 예측**하는 실험 코드입니다.

전체 그래프에서 노드의 도메인과 역할을 파싱한 뒤, `AI-MB` 서브그래프를 **source graph**, `AI-AE` 서브그래프를 **target graph**로 구성합니다. 이후 CompGCN 기반 인코더, domain adversarial training, hard negative mining을 활용해 타깃 도메인의 잠재적 관계를 예측하고, 최종 결과를 CSV 파일로 저장합니다.

---

## 주요 기능

- JSON 형식 지식 그래프 로드 및 전처리
- 노드 라벨에서 **도메인(AI, MB, AE)** 및 **역할(problem, solution)** 파싱
- `AI+MB` / `AI+AE` 서브그래프 자동 구성
- CompGCN 기반 노드 임베딩 학습
- 링크 존재 여부에 대한 binary classification
- GRL(Gradient Reversal Layer)을 이용한 domain adversarial training
- negative ratio, GRL 세기, entropy weight에 대한 grid search
- 관계 타입까지 함께 예측하는 joint classification
- 최종 예측 결과 CSV 저장

---

## 프로젝트 구조

이 코드는 하나의 스크립트 안에서 아래 순서로 실행되는 **실험 파이프라인형 코드**입니다.

1. 전체 그래프 로드
2. 도메인 및 역할 정보 파싱
3. source/target 서브그래프 생성
4. Stage 1: 링크 존재 여부 학습
5. Grid Search 수행
6. Stage 2: 관계 타입 포함 joint classification
7. 타깃 그래프 후보 링크 예측 및 CSV 저장

즉, 일반적인 `train.py` 하나짜리 프로젝트보다는 **노트북 또는 순차 실행형 실험 코드**에 가깝습니다.

---

## 입력 그래프 형식

기본 입력 파일은 아래와 같이 설정되어 있습니다.

```python
graph = "01.12_total.json"
```

코드는 다음 두 가지 형태의 그래프 JSON을 처리할 수 있습니다.

### 1) Triplet 스타일
- `n`: source node
- `r`: relation
- `m`: target node

### 2) 자유형 JSON 스타일
- node 키 예시: `id`, `name`, `label`, `labels`, `type`, `domain`, `role`
- edge 키 예시: `source`, `target`, `start`, `end`, `src`, `dst`, `type`, `relation`

노드 라벨은 문자열 또는 리스트 형태 모두 허용되며, 내부적으로 구분자를 기준으로 분해하여 도메인과 역할을 추출합니다.

---

## 도메인 및 역할 정의

### 도메인
- `AI` : Artificial Intelligence
- `MB` : Molecular Biology
- `AE` : Aerospace Engineering

### 노드 타입
- `problem`
- `solution`

이 정보는 노드 라벨에서 자동으로 파싱되며, 이후 서브그래프 구성과 평가 대상 edge 필터링에 사용됩니다.

---

## 서브그래프 구성

전체 그래프에서 두 개의 서브그래프를 생성합니다.

- **Source graph**: `AI + MB`
- **Target graph**: `AI + AE`

즉, AI와 MB 사이에서 학습한 관계 구조를 AI와 AE 영역으로 전이하는 것이 핵심 실험 목표입니다.

---

## 모델 구성

### 1. CompRGCNEncoder
CompGCN 레이어를 기반으로 노드 임베딩을 학습하는 인코더입니다.

- initial entity embedding
- initial relation embedding
- 2-layer CompGCN
- dropout 적용

### 2. PairClassifier
두 노드 사이에 링크가 존재하는지를 예측하는 binary classifier입니다.

입력 특징은 다음 네 가지를 결합합니다.

- `hu`
- `hv`
- `|hu - hv|`
- `hu * hv`

### 3. Domain Adversarial Head
GRL(Gradient Reversal Layer)을 사용하여 source와 target의 표현 차이를 줄입니다. 이를 통해 source domain에서 학습한 패턴이 target domain으로 더 잘 전이되도록 유도합니다.

### 4. JointHead
후반부에는 `No_Link` 클래스를 포함하는 multi-class 분류기를 사용하여, 링크 존재 여부뿐 아니라 **관계 타입까지 함께 예측**합니다.

---

## 학습 단계

## Stage 1. Binary Link Prediction
첫 번째 단계에서는 source graph를 사용하여 **링크 존재 여부**를 학습합니다.

주요 특징은 다음과 같습니다.

- domain-matched negative sampling
- GRL 기반 도메인 적대 학습
- entropy regularization
- early stopping 적용

학습 중 출력되는 주요 지표:

- Accuracy
- Balanced Accuracy
- AUC
- ACC@best threshold
- Hits@10

---

## Grid Search
다음 하이퍼파라미터 조합을 탐색합니다.

- `neg_ratio`
- `grl_lambda_max`
- `entropy_w`

타깃 그래프 기준 성능이 좋은 조합을 찾기 위해 여러 조합을 반복 실험합니다.

---

## Stage 2. Joint Relation Classification
두 번째 단계에서는 `JointHead`를 사용하여 다음을 함께 예측합니다.

- 링크가 존재하는가?
- 존재한다면 어떤 관계 타입인가?

이 단계에서는 multi-class 분류 및 relation ranking 기반 평가를 함께 수행합니다.

---

## Negative Sampling 전략

이 코드에서는 단순 랜덤 샘플링 대신 더 엄격한 negative sampling 전략을 사용합니다.

- **Domain-matched negative sampling**  
  양성 샘플의 tail 노드와 같은 도메인에서 음성 샘플을 추출합니다.

- **Hard negative mining**  
  모델이 헷갈려하는 어려운 음성 샘플을 선택합니다.

- **False negative filtering**  
  실제 양성 관계를 음성으로 잘못 뽑는 문제를 줄입니다.

이 설정은 보다 현실적인 링크 예측 평가에 도움이 됩니다.

---

## 주요 하이퍼파라미터

기본 설정은 다음과 같습니다.

```python
epochs = 200
lr = 3e-3
encoder_dim = 128
init_dim = 64
gcn_dim = 128
dropout = 0.1
entropy_w = 0.01
grl_lambda_max = 0.05
patience = 20
neg_ratio = 8
```

Grid Search 탐색 범위:

```python
search_neg_ratio  = [1, 5, 7, 8, 9]
search_grl_lambda = [0.01, 0.05, 0.10, 0.15, 0.20]
search_entropy_w  = [0.01, 0.02, 0.03, 0.04, 0.05]
```

---

## 평가 지표

### Binary Link Prediction
- Accuracy
- Balanced Accuracy
- AUC
- ACC@best threshold
- Hits@10

### Ranking Evaluation
- Hits@10
- MRR

### Joint Multi-class Evaluation
- Accuracy
- Macro AUC
- MRR
- Hits@10
- Micro F1
- Macro F1
- Weighted F1

즉, 이 프로젝트는 단순 분류 성능뿐 아니라 **랭킹 성능과 클래스 불균형 상황까지 함께 고려**합니다.

---

## 출력 파일

실행 결과는 CSV 형식으로 저장됩니다.

- `scopus_pred_edges_AE_Threshold.csv`  
  예측된 링크, 관계 타입, 링크 점수 등이 저장된 파일

- `01.20_pred_edges_named.csv`  
  원본 노드 identity와 이름 정보를 복원해 사람이 읽기 쉽게 정리한 파일

실험 후에는 위 파일을 기반으로 예측된 관계를 후속 분석에 사용할 수 있습니다.

---

## 실행 환경

다음 라이브러리가 필요합니다.

- Python
- PyTorch
- NumPy
- pandas
- 사용자 정의 CompGCN 구현
  - `model/compgcn_conv.py`

또한 코드 상단에서 GPU를 아래와 같이 직접 지정하고 있습니다.

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

실행 환경에 따라 해당 부분은 수정이 필요할 수 있습니다.

---

## 실행 방법

이 코드는 함수형 CLI 프로젝트라기보다는 **위에서 아래로 순차 실행하는 실험 스크립트**입니다.
따라서 일반적으로는 Jupyter Notebook 또는 순차 실행 가능한 Python 환경에서 사용하는 것이 적합합니다.

기본 실행 흐름은 다음과 같습니다.

```python
data, meta = load_graph_json(args.graph)
src = build_subgraph(data, keep_domains={AI, MB})
tgt = build_subgraph(data, keep_domains={AI, AE})

train_for_epochs(num_epochs=100)
results = run_grid_search()
train_stage1_joint()
```

이후 후반부 prediction/export 코드까지 실행하면 CSV 결과 파일이 생성됩니다.

---

## 주의사항

- 코드 내부 주석 일부에는 예전 실험 흔적으로 `Manu` 또는 `Manufacturing`라는 표현이 남아 있을 수 있습니다.
- 현재 실제 도메인 매핑은 `AI`, `MB`, `AE` 기준입니다.
- 입력 그래프의 라벨 형식이 일관되지 않으면 도메인/역할 파싱 결과가 달라질 수 있습니다.
- 이 코드는 하나의 통합 실험 스크립트이므로, 실제 프로젝트로 정리하려면 `data`, `train`, `eval`, `predict` 단계로 분리하는 것이 좋습니다.

---

## 개선 아이디어

- 실행 인자를 `argparse`로 분리하여 CLI 형태로 개선
- 데이터 로딩, 학습, 평가, 예측 코드를 별도 모듈로 분리
- 설정값을 YAML/JSON config 파일로 관리
- README에 예시 입력 그래프 구조 추가
- 결과 시각화 코드 별도 분리

---

## 요약

이 프로젝트는 다중 도메인 지식 그래프에서 **source domain의 관계 패턴을 target domain으로 전이**하기 위해 설계된 실험 코드입니다. CompGCN 기반 노드 표현 학습, domain adversarial alignment, hard negative mining, joint relation classification을 결합하여 단순 링크 예측을 넘어 **관계 타입까지 추론**할 수 있도록 구성되어 있습니다.
