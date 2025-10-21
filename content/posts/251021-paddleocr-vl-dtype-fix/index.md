+++
date = '2025-10-21T19:00:00+09:00'
title = 'PaddleOCR-VL dtype 오류 해결기 (WSL2 GPU)'
tags = ['PaddleOCR', 'AI', 'LLM', 'OCR']
categories = ['AI']

[[resources]]
  name = 'featured-image'
  src = 'Banner.png'
[[resources]]
  name = 'featured-image-preview'
  src = 'Banner.png'
+++

# PaddleOCR-VL dtype 오류 해결기 (WSL2 GPU)

WSL2 환경에서 PaddleOCR-VL을 GPU로 실행하려다 `Variable dtype not match` 에러와 씨름했습니다.  
CPU에서는 잘 돌아가는데 GPU로 전환하면 로딩 단계에서 바로 죽는다면, 이 글의 방법으로 해결할 수 있습니다.

---

## 1. 환경 정보

- **OS**: Windows 11 + WSL2 (Ubuntu 24.04)
- **Python**: venv (`.venv-paddleocrvl`) / Python 3.12
- **PaddlePaddle**: 3.2.0 (GPU 빌드)
- **PaddleOCR**: 3.3.0
- **safetensors**: 0.6.2.dev0 (paddle backend 지원)
- **GPU**: NVIDIA RTX 4060 Ti (Driver 581.29, CUDA 13.0)

> **Tip**  
> GPU 준비 상태는 다음으로 확인합니다.
> ```bash
> nvidia-smi
> ./.venv-paddleocrvl/bin/python -c "import paddle; print('CUDA build:', paddle.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count())"
> ```

---

## 2. 문제가 어떻게 발생했나?

GPU로 추론을 시작하면 바로 다음과 같은 에러가 발생했습니다.

```bash
AssertionError: Variable dtype not match, Variable [ layer_norm_0.w_0 ] need tensor with dtype paddle.bfloat16  but load tensor with dtype paddle.float32
```

PaddleOCR-VL은 Hugging Face 포맷의 safetensors 체크포인트를 받아 Paddle 모델로 변환합니다.  
이 과정에서 모델 파라미터는 **bfloat16**(GPU-friendly)로 초기화되는데, 가중치는 **float32**로 들어와서  
`set_state_dict` 단계에서 dtype assertion이 터지는 것이었습니다.

---

## 3. 1차 우회: CPU 실행

CPU에서는 bfloat16을 지원하지 않기 때문에 PaddleX 내부에서 자동으로 float32로 바꿔 주고,  
추론도 잘 됩니다. `--device cpu` 플래그로 확인할 수 있지만 GPU로는 그대로 실패했습니다.

---

## 4. 근본 해결: dtype 정렬 코드 수정

가상환경 안의 PaddleX 소스에서 dtype을 자동으로 맞춰주도록 수정했습니다.

수정 파일:  
`./.venv-paddleocrvl/lib/python3.12/site-packages/paddlex/inference/models/doc_vlm/modeling/paddleocr_vl/_paddleocr_vl.py`

핵심 패치 (하이라이트 부분만 표시):

```diff
@@ def set_hf_state_dict(self, state_dict, *args, **kwargs):
-        for key in std_state_dict:
-            v1 = std_state_dict[key]
-            state_dict[key] = state_dict[key].to(v1.place)
+        for key in std_state_dict:
+            v1 = std_state_dict[key]
+            tensor = state_dict[key]
+
+            # CPU처럼 bfloat16을 지원하지 않는 장치에서는 float32로 캐스팅
+            if (
+                tensor.dtype == paddle.bfloat16
+                and not paddle.amp.is_bfloat16_supported()
+            ):
+                tensor = tensor.astype(paddle.float32)
+
+            # 파라미터와 dtype이 다르면 맞춰 준다.
+            if tensor.dtype != v1.dtype:
+                tensor = tensor.astype(v1.dtype)
+
+            state_dict[key] = tensor.to(v1.place)

         return self.set_state_dict(state_dict, *args, **kwargs)
```

### 수정 방법

1. IDE 또는 `nano` 등으로 위 파일을 열어 해당 블록을 수정합니다.
2. 저장 후 GPU 추론을 다시 실행합니다.

> **주의**  
> `.venv` 내부 파일을 직접 수정한 것이므로 패키지를 재설치하면 수정 내용이 사라집니다.  
> PaddleX/PaddleOCR를 업데이트할 때는 다시 동일한 수정을 적용해야 합니다.

---

## 5. safetensors 버전 확인

`framework="paddle"` 옵션을 쓰려면 safetensors 0.4.3 이상이 필요합니다.  
WSL2에서 다음 명령으로 버전을 확인하고 필요하면 업데이트합니다.

```bash
./.venv-paddleocrvl/bin/pip show safetensors
./.venv-paddleocrvl/bin/pip install -U safetensors
```

---

## 6. GPU에서 재시도

```bash
./.venv-paddleocrvl/bin/paddleocr doc_parser \
    --device gpu:0 \
    --format_block_content true \
    --save_path ./ocr_outputs_gpu \
    -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png
```

이제는 dtype 에러 없이 추론과 결과 저장이 정상적으로 진행됩니다.

---

## 7. 마치며

- dtype 미스매치의 원인은 **모델 파라미터(bfloat16) vs 체크포인트(float32)** 불일치였습니다.
- PaddleX 내부에서 dtype을 파라미터에 맞춰 캐스팅하도록 패치하면 해결됩니다.
- safetensors 최신 버전을 사용해야 Hugging Face safetensors를 Paddle에서 읽을 수 있습니다.
- 추후 PaddleOCR/PaddleX 공식 패치가 나오면 이 수정을 제거하거나 맞춰 주세요.

