#!/usr/bin/env python3
"""추론 디바이스 선택 유틸.

CUDA를 요청했더라도 런타임에 쓸 수 없으면 죽지 않고 CPU로 내려간다.
panorama_stitcher의 CUDA 백엔드와 같은 정책이다 — 요청은 존중하되,
불가능하면 경고를 남기고 CPU 폴백. 노드가 통째로 죽는 것보다 느리게라도
차선이 나오는 편이 낫다.

CPU 폴백 시 FP16(half)도 반드시 함께 꺼야 한다. ultralytics는 CPU에서
half=True를 받으면 추론 중에 예외를 던진다.
"""
import torch


def resolve_device(logger, requested: str) -> tuple[str, bool]:
    """요청 디바이스를 실제 사용 가능한 값으로 해석한다.

    Args:
        logger: rclpy 노드 로거 (get_logger() 결과)
        requested: 'cuda:0', 'cuda', 'cpu' 등 파라미터로 받은 문자열

    Returns:
        (device, half) 튜플.
        device는 torch/ultralytics에 그대로 넘길 수 있는 문자열,
        half는 FP16 추론 가능 여부.
    """
    device = (requested or 'cpu').strip().lower()

    if not device.startswith('cuda'):
        logger.info(f'추론 디바이스: {device} (FP32)')
        return device, False

    if not torch.cuda.is_available():
        logger.warn(
            f"CUDA('{requested}')를 요청했지만 사용할 수 없습니다 "
            '(드라이버/런타임 미탑재 또는 GPU 미인식) — CPU로 폴백합니다')
        return 'cpu', False

    index = 0
    if ':' in device:
        suffix = device.split(':', 1)[1]
        if suffix.isdigit():
            index = int(suffix)
        else:
            logger.warn(f"디바이스 문자열 '{requested}' 해석 실패 — cuda:0을 사용합니다")

    count = torch.cuda.device_count()
    if index >= count:
        logger.warn(f'cuda:{index}를 요청했지만 GPU가 {count}개뿐입니다 — cuda:0으로 폴백합니다')
        index = 0

    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    logger.info(
        f'추론 디바이스: cuda:{index} ({torch.cuda.get_device_name(index)}, '
        f'여유 VRAM {free_bytes / 1024**3:.2f}/{total_bytes / 1024**3:.2f} GiB, FP16)')
    return f'cuda:{index}', True


def move_model_to_device(logger, model, device: str, half: bool) -> tuple[str, bool]:
    """모델을 디바이스로 올린다. 실패하면 CPU로 재시도한다.

    VRAM 부족(OOM)이나 드라이버 오류로 .to('cuda')가 던지는 경우를 잡는다.
    이 PC는 RTX 4060 8GB에 Xorg/gnome-shell이 이미 상주하고, 파노라마
    CUDA 백엔드와 동시에 뜨면 VRAM이 모자랄 수 있다.

    Returns:
        실제로 적용된 (device, half).
    """
    try:
        model.to(device)
        return device, half
    except Exception as exc:  # OOM, 드라이버 오류 등
        if device == 'cpu':
            raise
        logger.error(f"모델을 {device}로 올리지 못했습니다 ({exc}) — CPU로 폴백합니다")
        torch.cuda.empty_cache()
        model.to('cpu')
        return 'cpu', False
