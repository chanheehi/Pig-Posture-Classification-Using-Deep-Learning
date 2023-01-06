# Pig-Posture-Classification-Using-Deep-Learning
## 한국전자거래학회 학술대회 논문_참조 코드

## 가이드라인 
### 1. 데이터 셋 구성
**1.1 데이터를 VoTT를 활용해 라벨링하고, 파일명과 라벨명을 csv파일로 저장(./labeling/sample/label.csv)**  
데이터 셋은 AI HUB에서 제공하였고, ./labeling/sample에는 결과 재연을 위한 jpg 파일 1,000개로 구성됨  
**1.2 라벨링한 결과대로 label.csv파일을 활용해 각 클래스대로 정리(./labeling/class/[label]**  
라벨링 검수는 1차(육안), 2차(모델 예측)으로 이루어짐  
1차는 라벨링한 결과가 맞는지 육안으로 확인하였고, 2차는 만들어진 모델의 Test 셋에 대입해 검수하였음  
**1.3 검수를 통과한 데이터는 Train 셋에 추가**  
Train 셋과 Test 셋은 파일명의 Index 번호를 통해 철저히 구분하였음(ex. 6000번 이하: Train / 15000 이상: Test)  
### 2. 데이터 전처리&증대 ###  
**2.1 Sitting의 수가 상대적으로 적었으므로 ./Augmentation.py를 약간 수정해 데이터의 회전/이동/반전을 적용시켜 데이터를 증대**  
증대한 이미지는 구별할 수 있도록 파일명 앞 Aug_를 추가(ex. Aug_livestock_pig_keypoints_002717_0.jpg)    
**2.2 ./Augmentation.py에서 Cutout 및 데이터 변형을 적용**  
### 3. 모델링 ###  
**3.1 하이퍼파라미터를 설정**  
시연 결과 Best performance를 보인 하이퍼파라미터로 설정함  
### 4. 결과 ###  
**4.1 모델을 학습시키고 나온 결과를 통해 아래와 같은 결과를 확인할 수 있음**  
