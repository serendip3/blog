+++
title = 'Claude Dispatch와 Channels — OpenClaw에 대한 생각'
date = '2026-03-21T10:00:00+09:00'
draft = false
tags = ['Claude', 'Anthropic', 'Dispatch', 'Channels', 'OpenClaw', 'Agentic AI']
categories = ['Tech', 'AI']
description = 'Claude의 Dispatch와 Channels 기능, 그리고 OpenClaw 대한 생각'

[[resources]]
  name = 'featured-image'
  src = 'banner.jpg'

[[resources]]
  name = 'featured-image-preview'
  src = 'banner.jpg'
+++

# OpenClaw 열풍

OpenClaw 나오고 맥미니 품절 소식이 들렸고, 주변에도 맥미니 사서 돌려보는 사람이 꽤 있었어요.

OpenClaw가 사람들에게 인기가 있었던 건 뭘까 생각해 보면,

1. 에이전트 비서 같이 동작함 (메모리, 스케줄)
2. 메신저 같은 UI 로 대화 할 수 있음
3. 여러 도구들을 쉽게 붙일 수 있게 해 줌

이 세 가지를 비교적 쉽게 쓸 수 있었기 때문이라고 봐요.

# Claude 의 구독 요금제 우회 차단

OpenClaw는 토큰을 많이 쓰죠. 지인은 영화 취소표 예매 기능을 구현하라고 요청했는데, 매번 사이트를 방문해서 크롤링 혹은 캡처를 통해 진행했고, 제미나이 API를 설정했다가 하루 만에 4만 원이 나왔구요. 그래서 로컬 모델을 쓰고 싶어 하더라구요. 하지만 맥미니에서 구동할 수 있는 로컬 모델로는 만족하기 힘들죠.

API를 쓰는 건 개인에게 비용 부담이 크고, 구독 요금제를 그대로 쓰도록 연동했는데, 사실상 OpenAI를 제외하고는 모두 막는 상황이죠.

# OpenClaw 기능을 Claude 로 출시

1. **Claude Code의 remote-control** — 폰이나 브라우저에서 로컬 Claude Code 세션을 원격 조작
2. **Claude Cowork의 Dispatch** — 폰에서 작업을 지시하면 데스크톱 Cowork이 알아서 처리
3. **Claude Code의 Channels** — Telegram, Discord로 Claude Code 세션과 대화

OpenClaw가 보여준 핵심 니즈를 차근차근 Claude 내부로 흡수하고 있어요. Claude 모델을 좋아하는 사람들을 자사 서비스 안으로 락인하는 전략이죠.

저도 tmux 멀티 세션 관리 서비스 등을 개인적으로 만들어 쓰고 있지만, Claude를 쓸 때는 그냥 remote-control 하나 띄워 놓고 모바일로 처리해요.

# 프론티어로부터 도망칠 수 있는가?

결국 모델 성능이 핵심이고, 그걸 잘 쓸 수 있게 해주는 에이전틱 환경까지 플랫폼이 직접 깔아주고 있는데, 그걸 단지 wrap해서 편의성을 얹는 것만으로는 플랫폼이 직접 움직이는 순간 존재 가치가 빠르게 낮아져요. OpenClaw가 의미 없는 서비스는 아니지만, 초반의 센세이션을 프론티어가 하나씩 흡수하면서 입지가 좁아지고 있는 게 보이죠.

물론 OpenClaw 개발자는 큰 성공을 거두었죠. 하지만 일반적인 사람은 재미 혹은 기술을 먼저 해본다는 취미적 입장으로 접근해야 한다고 생각해요. 내가 만드는 것이 프론티어 모델이나 범용 솔루션에 의해 대체될 수 있는지 먼저 고민하고 리소스를 투입해야 해요. 개인이나 소규모 팀이라면 플랫폼과 정면으로 겹치는 영역보다는, 도메인 특화된 문제에 집중하는 편이 오래 살아남을 수 있는 길 같아요.


