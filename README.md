# RIFE for PotPlayer

[English](README.en.md)

**v0.2.0 · Windows x64 · RIFE 4.25 Standard**

**지원 GPU는 NVIDIA GeForce RTX 50 시리즈뿐입니다.** RTX 5090은 검증 완료, RTX 5080은 사용자 정상 구동 확인 상태입니다. RTX 30 시리즈는 사용자 테스트에서 동작하지 않아 지원하지 않습니다. 그 외 GPU 시리즈도 이번 배포본의 지원 대상이 아닙니다. 8K 60fps ×4 성능 검증은 RTX 5090 기준입니다.

RTX 5090에서 8K 60fps ×4 → 240fps를 시험한 팟플레이어용 RIFE 보간 필터입니다. 최대 긴 변 960에서 구한 움직임·혼합 마스크로 원본 해상도의 픽셀을 합성합니다.

[독립 실행 ZIP 다운로드](https://github.com/mkyoon80-alt/rife-potplayer/releases/tag/v0.2.0) · [한글 HTML 매뉴얼](https://mkyoon80-alt.github.io/rife-potplayer/) · [English HTML manual](https://mkyoon80-alt.github.io/rife-potplayer/manual.en.html)

ZIP 전체를 압축 해제한 뒤 **RifeControl.exe**를 실행하세요. .NET·Python·VapourSynth를 따로 설치할 필요가 없습니다. 팟플레이어 64bit와 호환 NVIDIA 드라이버는 별도로 필요합니다.

1. **연결 설정 → 엔진 검사 / 준비**
2. **필터 등록** (관리자 실행 팟플레이어는 관리자용 등록)
3. 팟플레이어 **F5 → 코덱/필터 → 전역 필터 우선 순위 → 시스템 코덱 추가**에서 **RIFE for PotPlayer**를 활성화하고 **최우선 사용** 선택
4. 영상을 다시 열고 **프레임 보간 동작 중**과 출력 FPS 확인

기본 출력 상한은 60fps입니다. 위 링크에서 매뉴얼을 바로 읽을 수 있습니다. 인터넷 없이 읽으려면 압축 해제한 폴더의 `docs/manual.ko.html` 또는 `docs/manual.en.html`을 브라우저로 여세요.

엔진은 PC 환경에 맞춰 한 번 준비하고 파일마다 재사용합니다. 재생·탐색 중에는 엔진을 컴파일하지 않습니다. 현재 배포본은 RTX 50 시리즈용 SM120 / PTX 빌더 리소스를 포함합니다. 지원 GPU에서 엔진 검사를 실행하고 필요하면 재준비하세요. 엔진 재준비가 RTX 30 등 다른 시리즈의 지원을 보장하지는 않습니다.

초기 원본 준비는 실제시간 100ms, 탐색 직후 원본 표시 기준은 50ms입니다. 상세 진단은 32쌍마다 표본을 기록합니다. 재생 중 진단 텍스트 갱신으로 스크롤과 선택 영역이 초기화되던 문제를 수정했습니다.

8K 실시간 시험은 모든 영상·PC에서 무드랍을 보장하는 뜻이 아닙니다. 남은 초기 드랍과 VSR 동시 사용 시 GPU 경합은 장기 개선 과제입니다. Dolby Vision은 메타데이터 전달을 검증한 범위이며 모든 프로파일·FEL·디스플레이 출력을 보증하지 않습니다. HDR10+ 동적 메타데이터 전달은 미구현입니다.

[빌드 안내](docs/BUILDING.md) · [변경 사항](CHANGELOG.md) · [라이선스 고지](THIRD_PARTY_NOTICES.md)
