# MachineLearningWorkflow

## Acquisition
- 조사나 연구 목적에 의해 특정 도메인으로부터 수집된 텍스트 집합을 코퍼스(corpus)라고 부름
- 코퍼스를 긁어모으는 단계

## Inspection & Exploration
- 긁어모은 데이터에 대한 구조, 노이즈 데이터, 데이터 정제 방법 등을 파악하는 단계
- 이 과정을 EDA (Exploratary Data Analysis, 탐색적 데이터 분석)이라고 부르기도 한다.
    * 독립 변수, 종속 변수, 변수 데이터 타입, 변수 유형 등을 점검하여 특징과 내재 구조 관계를 알아내는 과정

## Preprocessing & Cleaning
- 토큰화, 정제, 정규화, 불용어 제거 등과 같은 데이터 전처리 과정

## Data Split
- 학습, 검증, 평가 데이터 등으로 나누는 과정

## Modeling
- 머신 러닝 모델의 구조를 작성하는 과정

## Training
- 모델에 데이터를 학습시키는 과정

## Evaluation
- 학습된 모델을 평가하는 과정
- 만족스럽지 못한 경우에는 이전 단계로 돌아감

## Deployment
- 평가 과정까지 무사히 넘겼다면 배포하는 과정
- 업데이트 상황이 발생할 경우에는 다시 수집 과정으로 돌아감