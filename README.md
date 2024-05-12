

# Fake_Guard

# 프로젝트 개요
최근 급격한 생성형 AI의 발전으로 그에 따른 사회적 문제 또한 대두된다.
그중 가장 큰 문제는 유명인이나 일반인의 얼굴을 이용하여 생성한 딥페이크라고 불리는 가짜영상이다.


https://github.com/lee-hyeon-gu/Fake_Guard/assets/149747454/b1a3c5cc-9aa5-4f1f-8387-5449167907a5


우리는 그에 대한 해결책으로 AI를 이용한 자동 탐지 시스템을 제작하고자 하였다.

딥페이크 자료의 경우 이미지만을 조작한 영상, 생성된 사운드를 덧입힌 영상, 두가지 모두 조작된 영상의 세가지의 경우가 있다.
대표적인 조작방식은 이미지 조작형식이지만 다른 방식에 대해서도 참조를 하면 조작영상 판단에 도움이 될 것이라 생각하여 모든 경우에 대해 고려해보기로 하였다.

# 추론 방식
동영상을 업로드하면
FastAPI를 MySQL과 연동하여 URL과 동영상 이름을 저장하고
동영상은 TALL++모델의 프레임 분석
이미지는 EfficientNet
사운드는 SpecRNet 을 이용하여 각각 분석을 하였다.

# 작업 흐름도(전처리)

[이미지]
   
![이미지 전처리](https://github.com/lee-hyeon-gu/Fake_Guard/assets/149747454/50aa15b9-a095-420c-8c5a-b5b0bb8ae515)

[오디오]
   
![오디오 전처리](https://github.com/lee-hyeon-gu/Fake_Guard/assets/149747454/65edb264-7413-4415-b9dc-3c497f52afb5)

# 작업 흐름도(모델)

[동영상]

![동영상처리](https://github.com/lee-hyeon-gu/Fake_Guard/assets/149747454/5f73961e-c2ec-4edc-9a2d-3d318c7d5d56)

[이미지]

![이미지 처리](https://github.com/lee-hyeon-gu/Fake_Guard/assets/149747454/727732e7-2d98-4a63-ba79-1bdb3304deb3)

# 시연 영상


https://github.com/lee-hyeon-gu/Fake_Guard/assets/149747454/e0c94a14-bf9b-4482-83c3-e237ae93a3df


# Reference 

# Related and additional tools
1. TALL++ https://github.com/rainy-xu/TALL4Deepfake
