# [Paper Review] Generative Language Models for Paragraph-Level Question Generation

> Ushio, A., Alva-Manchego, F., & Camacho-Collados, J. (2022). Generative Language Models for Paragraph-Level Question Generation. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*.

---

## 1. Introduction

### 문제 제기
* **임의적인(arbitrary) 모델 선택**: 그동안 질문 생성(Question Generation, QG) 연구는 표준화된 벤치마크가 부재하여 모델 선택과 평가가 체계적이지 못하고 임의로 이루어지는 경향이 있었습니다.
* **평가 지표의 한계**: BLEU, METEOR, ROUGE-L 같은 자동 평가 지표는 사람이 만든 정답 질문과 비교하는 방식으로, '답변 가능성'이나 '품질'에 대한 사람의 실제 판단과는 상관관계가 낮다는 문제가 있었습니다.
* **QG Task의 파편화**: QG Task는 `sentence-level` vs `paragraph-level`, `answer-aware` vs `answer-free` 등 종류가 너무 다양해 통합적인 평가가 어려웠습니다.

### 해결 방안 제시
* **QG-Bench 제안**: 이러한 문제들을 해결하기 위해 8개의 다른 언어와 다양한 도메인으로 구성된 새로운 통합 벤치마크 **`QG-Bench`**를 제안합니다.
* **연구 접근법**: Paragraph-Level QG Task를 Sequence-to-Sequence 생성 과제로 정의하고, 제안한 벤치마크를 통해 도메인별, 언어별로 모델 성능을 심도 있게 측정 및 분석합니다.

---

## 2. Related Work

* **QG 연구의 흐름**: 초기 파이프라인 접근법에서 신경망 기반의 End-to-End `seq2seq` 모델로 발전했으며, 사전 훈련된 언어 모델을 파인튜닝하는 방식이 SOTA(State-of-the-Art)로 자리 잡았습니다.
* **기존 벤치마크와의 차별점**:
  * 유사한 벤치마크로 `MTG`가 있었지만, QG Task의 비중이 너무 적다는 한계가 있었습니다.
  * `QG-Bench`는 QG에 더 집중하며, 영어-타언어 **번역 쌍 데이터가 아닌** 각 언어 고유의 **독립된 데이터(monolingual data)**를 활용하여 보다 자연스러운 언어 환경을 반영합니다. 또한, 다양한 "도메인"과 "스타일"을 포함하여 포괄적인 평가를 가능하게 합니다.

---

## 3. QG-Bench: A Unified Question Generation Benchmark

### 3.1 Data Collection and Unification

* **데이터 구조**: `paragraph`, `sentence`, `question`, `answer`의 4가지 주요 특성으로 데이터를 통일했습니다.
* **포함된 데이터셋**:
  * **SQuAD**: 위키피디아 기반의 표준 데이터셋
  * **SQuADShifts**: Amazon, Wiki, News, Reddit 등 다양한 도메인 포함
  * **SubjQA**: 책, 전자제품, 식료품, 영화 등 6개 도메인에 걸친 주관적인 질문 포함
  * **다국어 데이터셋**: `JAQuAD`(일본어), `GerQuAD`(독일어), `SberQuAD`(러시아어), **`KorQuAD`**(한국어) 등
* **제외된 데이터셋**: `BioASQ`, `NewsQA` 등은 지문의 길이가 다른 데이터셋과 맞지 않아 제외했으며, `XQuAD`, `MLQA` 등은 train set이 없어 제외했습니다.

### 3.2 Data Statistics

* 각 데이터셋의 크기, 언어, 도메인 등에 대한 통계는 논문의 표에 상세히 기술되어 있습니다.

---

## 4. LMs for Question Generation

### 4.1 Task Formulation

* QG Task를 조건부 시퀀스 생성(Conditional Sequence Generation) 과제로 정의합니다.
* 즉, 입력 텍스트 `c`가 주어졌을 때, 생성된 질문 `q`의 확률 `P(q|c)`를 최대화하는 `q_hat`을 찾는 문제입니다.

### 4.2 Language Model Fine-tuning

* T5, BART와 같은 seq2seq 모델을 파인튜닝하는 표준적인 방식을 따릅니다.
* 입력 텍스트(paragraph) 내에서 정답(answer)에 해당하는 부분은 특별 토큰(special token)으로 감싸 **강조(highlight)**하는 방식을 사용합니다.

### 4.3 Experimental Setup

* **사용 모델**: `T5`, `BART`, `mT5`, `mBART`
* **하이퍼파라미터 튜닝**: 최적의 하이퍼파라미터 조합을 찾기 위해 광범위한 탐색을 수행했습니다. BLEU4 점수를 기준으로 최적 모델을 선택했으며, 이는 계산이 가볍고 이전 연구들과의 비교에 용이하기 때문이라고 밝혔습니다.

---

## 5. Automatic Evaluation

### 5.1 Evaluation Metrics

* 전통적인 지표 외에, 사람의 판단과 더 높은 상관관계를 보인다고 알려진 **`BERTScore`**와 **`MoverScore`**를 추가로 사용하여 평가의 신뢰도를 높였습니다.

### 5.2 Results

* 평가는 크게 3가지 관점에서 수행되었습니다.
  1.  **SQuAD**: 표준 SQuAD 데이터셋에서의 성능
  2.  **Language-Specific**: 언어별 데이터셋(한국어 포함)에서의 성능
  3.  **Domain-Specific**: 특정 도메인 데이터셋에서의 성능
* **한국어 데이터(KorQuAD) 성능**: mT5-BASE 모델이 준수한 성능을 보였으며, 이는 향후 연구를 통해 충분히 개선 및 경쟁이 가능한 수준으로 판단됩니다.

---

## 6. Analysis

### 6.1 Model Input

* **입력 방식 비교**: `paragraph-level`(전체 문단 입력) 방식이 `sentence-level`(문장만 입력) 방식보다 항상 더 좋은 성능을 보였습니다. 이는 모델이 실제로 지문의 **전체적인 문맥(global context)**을 효과적으로 활용하고 있음을 시사합니다.

### 6.2 Manual Evaluation

* **수동 평가 도입**: 자동 평가 지표의 한계를 보완하기 위해 Amazon Mechanical Turk (AMT)를 통해 사람이 직접 평가를 수행했습니다.
* **평가 기준**: `문법성(Grammaticality)`, `이해 가능성(Understandability)`, `답변 가능성(Answerability)`의 3가지 기준을 사용했습니다.
* **평가자 간 신뢰도 (Fleiss's Kappa)**:
  * **`답변 가능성`** 항목은 카파(κ) 값이 0.61로 높아 평가자 간 판단이 **상당히 일치**했고, 신뢰할 수 있는 평가임을 확인했습니다.
  * 반면, `문법성`(0.30)과 `이해 가능성`(0.36)은 카파 값이 낮아, 평가자마다 기준이 다소 주관적이었음을 알 수 있습니다. 이는 향후 QG 평가 연구에서 고려해볼 만한 중요한 포인트입니다.
* **핵심 통계 기법**: 논문의 신뢰도를 높이기 위해 아래와 같은 통계적 방법론들이 사용되었습니다.
  * **Fleiss's Kappa**: 3명 이상 평가자 간의 신뢰도(일치도) 측정.
  * **Spearman’s Rank Correlation**: 자동 평가 순위와 사람의 평가 순위 간의 관련성 측정.
  * **William (Wilcoxon) Test**: 두 조건/지표 간의 차이가 통계적으로 유의미한지 검증.

---

## Appendix

* 부록에는 각종 통계 실험의 상세 내용과 질문 평가 가이드라인(Question Evaluation Guideline) 등이 포함되어 있습니다.