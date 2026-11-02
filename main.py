"""
MyVideo AI - Локальний сервер генерації відео (FastAPI + FFmpeg + Edge-TTS)
Підтримує локальний безкоштовний рендеринг:
- Image-to-Video (Ken Burns з плавними переходами, кольорокорекцією, віньєтуванням)
- Text-to-Video (процедурний анімований градієнт з рухомими частинками через numpy/PIL)
- Озвучення українською мовою через Edge-TTS або процедурний ембієнт-звук
- Title-картка з текстом промпту (2 сек)
- Водяний знак: 'ТЕСТОВИЙ РЕЖИМ (FREE)'
- Заглушка для майбутнього Google GenAI SDK (Veo)
"""

import asyncio
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

# -------------------------------------------------------------------------
# ПЛАТНА AI-ГЕНЕРАЦІЯ (fal.ai — Bytedance Seedance 2.5, офіційний REST API)
# Документація: https://fal.ai/models/bytedance/seedance-2.5/image-to-video
#               https://fal.ai/models/bytedance/seedance-2.5/text-to-video
# Вмикається встановленням змінної середовища FAL_API_KEY. Якщо її немає —
# сервер автоматично працює у безкоштовному локальному тестовому режимі.
# -------------------------------------------------------------------------
FAL_API_KEY = os.environ.get("FAL_API_KEY", "").strip()
FAL_IMG2VID_URL = "https://fal.run/bytedance/seedance-2.5/image-to-video"
FAL_TXT2VID_URL = "https://fal.run/bytedance/seedance-2.5/text-to-video"

# -------------------------------------------------------------------------
# ПЛАТНА AI-ГЕНЕРАЦІЯ (kie.ai — Bytedance Seedance 2.5, офіційний REST API)
# Документація: https://kie.ai/seedance-2-5 (вкладка "API")
# Вмикається встановленням змінної середовища KIE_API_KEY.
# Якщо задані ОБИДВА ключі (FAL_API_KEY і KIE_API_KEY) — пріоритет у kie.ai,
# бо він дешевший. Якщо немає жодного — безкоштовний локальний режим.
# -------------------------------------------------------------------------
KIE_API_KEY = os.environ.get("KIE_API_KEY", "").strip()
KIE_CREATE_TASK_URL = "https://api.kie.ai/api/v1/jobs/createTask"
KIE_RECORD_INFO_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"
KIE_MODEL_NAME = "bytedance/seedance-2.5"

# Render автоматично задає RENDER_EXTERNAL_URL — публічну адресу сервера.
# Вона потрібна, щоб fal.ai/kie.ai могли завантажити надіслане користувачем фото.
PUBLIC_BASE_URL = (
    os.environ.get("PUBLIC_BASE_URL")
    or os.environ.get("RENDER_EXTERNAL_URL")
    or f"http://localhost:{os.environ.get('PORT', 8000)}"
).rstrip("/")

# Налаштування логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("myvideo_ai")

# Робочі каталоги
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
VIDEOS_DIR = STORAGE_DIR / "videos"
THUMBNAILS_DIR = STORAGE_DIR / "thumbnails"
TEMP_DIR = STORAGE_DIR / "temp"
HISTORY_FILE = STORAGE_DIR / "history.json"

for d in [VIDEOS_DIR, THUMBNAILS_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="MyVideo AI Server",
    description="Локальний бекенд генерації відео для додатку MyVideo AI",
    version="1.0.0"
)

# CORS для доступу з емулятора та локальної мережі Wi-Fi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Стан завдань у пам'яті
tasks = {}

# ----------------- БЕЗПЕКА (Path Traversal Protection) -----------------
SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")

def validate_safe_id(file_id: str, base_dir: Path, suffix: str = "") -> Path:
    # file_id має бути ЧИСТИМ ідентифікатором (без розширення) — суфікс (.mp4/.jpg)
    # додається окремо, інакше крапка в імені файлу завжди провалює SAFE_ID_REGEX
    # і ендпоінт помилково повертає 400 Bad Request навіть для коректних id.
    if not file_id or not SAFE_ID_REGEX.match(file_id):
        raise HTTPException(status_code=400, detail="Некоректний ідентифікатор файлу")
    target_path = (base_dir / f"{file_id}{suffix}").resolve()
    base_resolved = base_dir.resolve()
    if not str(target_path).startswith(str(base_resolved)):
        raise HTTPException(status_code=400, detail="Спроба несанкціонованого доступу до шляху (Path Traversal)")
    return target_path

# ----------------- УПРАВЛІННЯ ІСТОРІЄЮ -----------------
def load_history() -> List[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Помилка зчитування history.json: {e}")
        return []

def save_history_item(item: dict):
    history = load_history()
    # Видаляємо дублікати за video_id якщо є
    history = [h for h in history if h.get("video_id") != item.get("video_id")]
    history.insert(0, item)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Помилка запису в history.json: {e}")

def delete_history_item(video_id: str):
    history = load_history()
    history = [h for h in history if h.get("video_id") != video_id]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Помилка оновлення history.json при видаленні: {e}")

# ----------------- СТРУКТУРИ ДАНИХ (Pydantic) -----------------
class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0 to 100
    message: str
    video_id: Optional[str] = None
    error: Optional[str] = None
    created_at: float

class EditRequest(BaseModel):
    video_id: str
    effect: str  # "speed_0.5", "speed_1.5", "speed_2.0", "vignette", "vintage", "cinema", "bw"
    title: Optional[str] = None

# =========================================================================
# ПЛАТНА AI-ГЕНЕРАЦІЯ (fal.ai / Bytedance Seedance 2.5)
# =========================================================================
def generate_video_via_fal(
    task_id: str,
    prompt: str,
    aspect_ratio: str,
    duration: int,
    image_path: Optional[str],
    video_id: str,
    final_video_path: Path,
    final_thumb_path: Path,
):
    """Справжня генерація відео через офіційний REST API fal.ai (Seedance 2.5)."""
    headers = {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
    }

    if image_path and os.path.exists(image_path):
        tasks[task_id]["message"] = "Завантаження фото для Seedance 2.5 (Image-to-Video)..."
        image_url = f"{PUBLIC_BASE_URL}/uploads/{Path(image_path).name}"
        payload = {
            "image_url": image_url,
            "prompt": prompt,
            "resolution": "720p",
            "duration": str(duration),
            "generate_audio": True,
        }
        endpoint = FAL_IMG2VID_URL
    else:
        tasks[task_id]["message"] = "Генерація відео через Seedance 2.5 (Text-to-Video)..."
        payload = {
            "prompt": prompt,
            "resolution": "720p",
            "duration": str(duration),
            "aspect_ratio": aspect_ratio,
            "generate_audio": True,
        }
        endpoint = FAL_TXT2VID_URL

    tasks[task_id]["progress"] = 30
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=600)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Не вдалося з'єднатись з fal.ai: {e}")

    if response.status_code != 200:
        # Пробуємо дістати зрозумілий текст помилки від fal.ai (напр. брак коштів)
        try:
            err_detail = response.json()
        except Exception:
            err_detail = response.text
        raise RuntimeError(f"fal.ai повернув помилку {response.status_code}: {err_detail}")

    data = response.json()
    video_info = data.get("video")
    if not video_info or not video_info.get("url"):
        raise RuntimeError(f"fal.ai не повернув відео у відповіді: {data}")

    video_remote_url = video_info["url"]

    tasks[task_id]["progress"] = 85
    tasks[task_id]["message"] = "Завантаження готового відео з fal.ai..."
    with requests.get(video_remote_url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(final_video_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)

    # Мініатюра з готового відео
    tasks[task_id]["progress"] = 95
    tasks[task_id]["message"] = "Створення мініатюри..."
    thumb_cmd = [
        "ffmpeg", "-y", "-ss", "00:00:01", "-i", str(final_video_path),
        "-vframes", "1", "-q:v", "2", str(final_thumb_path)
    ]
    res = subprocess.run(thumb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not final_thumb_path.exists() or final_thumb_path.stat().st_size == 0:
        subprocess.run([
            "ffmpeg", "-y", "-ss", "00:00:00.2", "-i", str(final_video_path),
            "-vframes", "1", "-q:v", "2", str(final_thumb_path)
        ], check=True)

# =========================================================================
# ПЛАТНА AI-ГЕНЕРАЦІЯ (kie.ai / Bytedance Seedance 2.5)
# =========================================================================
def generate_video_via_kie(
    task_id: str,
    prompt: str,
    aspect_ratio: str,
    duration: int,
    image_path: Optional[str],
    video_id: str,
    final_video_path: Path,
    final_thumb_path: Path,
):
    """Справжня генерація відео через офіційний REST API kie.ai (Seedance 2.5)."""
    headers = {
        "Authorization": f"Bearer {KIE_API_KEY}",
        "Content-Type": "application/json",
    }

    input_data = {
        "prompt": prompt,
        "resolution": "720p",
        "aspect_ratio": aspect_ratio if not (image_path and os.path.exists(image_path)) else "adaptive",
        "duration": duration,
        "generate_audio": True,
        "output_format": "mp4",
    }

    if image_path and os.path.exists(image_path):
        tasks[task_id]["message"] = "Завантаження фото для Seedance 2.5 (kie.ai)..."
        image_url = f"{PUBLIC_BASE_URL}/uploads/{Path(image_path).name}"
        input_data["first_frame_url"] = image_url
    else:
        tasks[task_id]["message"] = "Генерація відео через Seedance 2.5 (kie.ai)..."

    body = {
        "model": KIE_MODEL_NAME,
        "input": input_data,
    }

    tasks[task_id]["progress"] = 30
    try:
        response = requests.post(KIE_CREATE_TASK_URL, headers=headers, json=body, timeout=60)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Не вдалося з'єднатись з kie.ai: {e}")

    if response.status_code != 200:
        raise RuntimeError(f"kie.ai повернув помилку {response.status_code}: {response.text}")

    resp_json = response.json()
    if resp_json.get("code") != 200:
        raise RuntimeError(f"kie.ai відхилив завдання: {resp_json.get('msg')}")

    kie_task_id = resp_json["data"]["taskId"]
    logger.info(f"kie.ai завдання створено: {kie_task_id}")

    # Опитування статусу (поки немає публічного callBackUrl — просто polling)
    tasks[task_id]["message"] = "Очікування генерації відео на kie.ai (може тривати кілька хвилин)..."
    max_wait_seconds = 15 * 60
    poll_interval = 6
    elapsed = 0
    result_urls = None

    while elapsed < max_wait_seconds:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            status_resp = requests.get(
                KIE_RECORD_INFO_URL,
                headers=headers,
                params={"taskId": kie_task_id},
                timeout=30,
            )
        except requests.exceptions.RequestException:
            continue  # тимчасовий збій мережі — пробуємо ще раз на наступній ітерації

        if status_resp.status_code != 200:
            continue

        status_data = status_resp.json().get("data", {})
        state = status_data.get("state")

        progress_by_state = {"waiting": 35, "queuing": 45, "generating": 65}
        if state in progress_by_state:
            tasks[task_id]["progress"] = progress_by_state[state]

        if state == "success":
            result_json_raw = status_data.get("resultJson")
            if not result_json_raw:
                raise RuntimeError("kie.ai позначив завдання успішним, але не повернув resultJson")
            result_data = json.loads(result_json_raw)
            result_urls = result_data.get("resultUrls")
            if not result_urls:
                raise RuntimeError(f"kie.ai не повернув відео у resultUrls: {result_data}")
            break
        elif state == "fail":
            fail_msg = status_data.get("failMsg") or status_data.get("failCode") or "невідома помилка"
            raise RuntimeError(f"kie.ai не зміг згенерувати відео: {fail_msg}")

    if not result_urls:
        raise RuntimeError("Час очікування вичерпано — kie.ai не завершив генерацію за 15 хвилин")

    video_remote_url = result_urls[0]

    tasks[task_id]["progress"] = 90
    tasks[task_id]["message"] = "Завантаження готового відео з kie.ai..."
    with requests.get(video_remote_url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(final_video_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)

    tasks[task_id]["progress"] = 95
    tasks[task_id]["message"] = "Створення мініатюри..."
    thumb_cmd = [
        "ffmpeg", "-y", "-ss", "00:00:01", "-i", str(final_video_path),
        "-vframes", "1", "-q:v", "2", str(final_thumb_path)
    ]
    subprocess.run(thumb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not final_thumb_path.exists() or final_thumb_path.stat().st_size == 0:
        subprocess.run([
            "ffmpeg", "-y", "-ss", "00:00:00.2", "-i", str(final_video_path),
            "-vframes", "1", "-q:v", "2", str(final_thumb_path)
        ], check=True)

# =========================================================================
# БЕЗКОШТОВНИЙ ЛОКАЛЬНИЙ РЕНДЕРИНГ (FFmpeg + Edge-TTS + NumPy/Pillow)
# =========================================================================

def get_dimensions(aspect_ratio: str):
    if aspect_ratio == "9:16":
        return 720, 1280
    elif aspect_ratio == "1:1":
        return 720, 720
    else:  # "16:9" за замовчуванням
        return 1280, 720

async def generate_speech_edge_tts(text: str, output_path: Path, voice: str = "uk-UA-PolinaNeural") -> bool:
    """Генерація українського озвучення через edge-tts"""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
    except Exception as e:
        logger.warning(f"Edge-TTS не зміг створити аудіо ({e}), перемикання на синтезований ембієнт-звук")
    return False

def generate_procedural_background_frames(
    output_dir: Path,
    width: int,
    height: int,
    total_frames: int,
    fps: int,
    prompt: str
):
    """
    Text-to-Video: процедурний анімований космічний градієнт з рухомими частинками
    (через numpy та PIL, НЕ testsrc2).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Створюємо частинки з фіксованим random seed
    np.random.seed(42)
    num_particles = 70
    particles = []
    for _ in range(num_particles):
        particles.append({
            "x": np.random.uniform(0, width),
            "y": np.random.uniform(0, height),
            "vx": np.random.uniform(-1.5, 1.5),
            "vy": np.random.uniform(-2.5, -0.5),  # пливуть вгору
            "radius": np.random.uniform(2.5, 8.0),
            "alpha": np.random.uniform(0.3, 0.9),
            "hue_shift": np.random.uniform(0, 1.0)
        })

    # Базова сітка для градієнта
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Визначаємо палітру (електричний фіолетовий -> глибокий синій -> неоновий циан)
    for f in range(total_frames):
        t = f / fps
        
        # Рухомі центри градієнта
        cx1 = width * (0.5 + 0.3 * math.sin(t * 0.8))
        cy1 = height * (0.5 + 0.3 * math.cos(t * 0.7))
        dist1 = np.sqrt((x_coords - cx1)**2 + (y_coords - cy1)**2) / (math.sqrt(width**2 + height**2) * 0.7)
        dist1 = np.clip(dist1, 0, 1)

        cx2 = width * (0.5 + 0.3 * math.cos(t * 1.1))
        cy2 = height * (0.5 + 0.3 * math.sin(t * 0.9))
        dist2 = np.sqrt((x_coords - cx2)**2 + (y_coords - cy2)**2) / (math.sqrt(width**2 + height**2) * 0.6)
        dist2 = np.clip(dist2, 0, 1)

        # Кольорові хвилі
        r = (25 + 75 * (1 - dist1) + 40 * math.sin(t * 1.5)).clip(15, 230).astype(np.uint8)
        g = (15 + 60 * (1 - dist2) + 50 * math.cos(t * 1.2)).clip(10, 200).astype(np.uint8)
        b = (60 + 140 * (1 - dist1 * 0.5 + 1 - dist2 * 0.5)).clip(40, 255).astype(np.uint8)
        
        img_arr = np.dstack((r, g, b))
        frame_img = Image.fromarray(img_arr, mode="RGB")
        draw = ImageDraw.Draw(frame_img, "RGBA")
        
        # Малюємо рухомі частинки зі свіченням
        for p in particles:
            px = (p["x"] + p["vx"] * f * 2) % width
            py = (p["y"] + p["vy"] * f * 2) % height
            rad = p["radius"] * (1.0 + 0.2 * math.sin(t * 3.0 + p["hue_shift"] * 6))
            alpha = int(255 * p["alpha"] * (0.7 + 0.3 * math.sin(t * 2.0 + p["hue_shift"] * 4)))
            
            # Зовнішній ореол свічення
            draw.ellipse(
                [px - rad * 1.8, py - rad * 1.8, px + rad * 1.8, py + rad * 1.8],
                fill=(120, 200, 255, int(alpha * 0.35))
            )
            # Ядро
            draw.ellipse(
                [px - rad, py - rad, px + rad, py + rad],
                fill=(240, 250, 255, alpha)
            )

        frame_path = output_dir / f"frame_{f:04d}.png"
        frame_img.save(frame_path, "PNG")

def create_title_card(prompt: str, width: int, height: int, output_path: Path):
    """Створює стильну 2-секундну заставку (title-картку) з текстом промпту"""
    img = Image.new("RGBA", (width, height), (15, 12, 32, 255))
    draw = ImageDraw.Draw(img)

    # Декоративні елементи та рамка
    margin = 36
    draw.rectangle(
        [margin, margin, width - margin, height - margin],
        outline=(110, 80, 230, 180),
        width=3
    )
    
    # Логотип/назва
    badge_text = "✦ MyVideo AI Studio ✦"
    try:
        font_badge = ImageFont.load_default()
    except Exception:
        font_badge = None

    draw.text((margin + 24, margin + 24), badge_text, fill=(0, 230, 255, 230), font=font_badge)

    # Текст промпту з автоперенесенням
    words = prompt.split()
    lines = []
    current_line = []
    max_chars_per_line = max(18, width // 28)
    for w in words:
        if len(" ".join(current_line + [w])) <= max_chars_per_line:
            current_line.append(w)
        else:
            lines.append(" ".join(current_line))
            current_line = [w]
    if current_line:
        lines.append(" ".join(current_line))

    # Відображення тексту
    start_y = height // 2 - (len(lines) * 24) // 2
    for i, line in enumerate(lines[:6]):
        draw.text(
            (width // 2 - (len(line) * 4), start_y + i * 28),
            line,
            fill=(255, 255, 255, 240),
            font=font_badge
        )

    # Водяний знак у кутку
    watermark_text = "ТЕСТОВИЙ РЕЖИМ (FREE)"
    draw.text(
        (width - margin - 190, height - margin - 28),
        watermark_text,
        fill=(255, 200, 100, 200),
        font=font_badge
    )

    img.convert("RGB").save(output_path, "JPEG", quality=92)

def generate_video_task(
    task_id: str,
    prompt: str,
    aspect_ratio: str,
    duration: int,
    image_path: Optional[str] = None
):
    """Головний воркер генерації відео (виконується у фоні)"""
    work_dir = TEMP_DIR / task_id
    work_dir.mkdir(parents=True, exist_ok=True)
    video_id = str(uuid.uuid4())[:12]
    final_video_path = VIDEOS_DIR / f"{video_id}.mp4"
    final_thumb_path = THUMBNAILS_DIR / f"{video_id}.jpg"

    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 10
        tasks[task_id]["message"] = "Підготовка параметрів відео..."

        # Якщо на сервері налаштовано KIE_API_KEY або FAL_API_KEY — використовуємо
        # справжню AI-генерацію (Seedance 2.5) замість локального рендеру.
        # kie.ai має пріоритет, якщо задано обидва ключі (дешевший тариф).
        paid_provider = None
        if KIE_API_KEY:
            paid_provider = "kie"
        elif FAL_API_KEY:
            paid_provider = "fal"

        if paid_provider:
            if paid_provider == "kie":
                generate_video_via_kie(
                    task_id, prompt, aspect_ratio, duration, image_path,
                    video_id, final_video_path, final_thumb_path
                )
                mode_label = "paid_kie_seedance"
            else:
                generate_video_via_fal(
                    task_id, prompt, aspect_ratio, duration, image_path,
                    video_id, final_video_path, final_thumb_path
                )
                mode_label = "paid_fal_seedance"

            history_item = {
                "video_id": video_id,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "has_image": bool(image_path),
                "mode": mode_label,
                "created_at": time.time(),
                "video_url": f"/video/{video_id}",
                "thumbnail_url": f"/thumbnail/{video_id}"
            }
            save_history_item(history_item)

            tasks[task_id]["status"] = "completed"
            tasks[task_id]["progress"] = 100
            tasks[task_id]["message"] = "Відео успішно створено (Seedance 2.5)!"
            tasks[task_id]["video_id"] = video_id
            logger.info(f"Завдання {task_id} ({paid_provider}) успішно завершено: {video_id}")
            return

        width, height = get_dimensions(aspect_ratio)
        fps = 24

        # 1. Створення аудіо (Edge-TTS або процедурний синтез тону)
        tasks[task_id]["progress"] = 25
        tasks[task_id]["message"] = "Генерація звукової доріжки (uk-UA)..."
        audio_path = work_dir / "voice.mp3"
        
        # Шукаємо діалог або використовуємо сам промпт
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        has_voice = loop.run_until_complete(
            generate_speech_edge_tts(prompt, audio_path)
        )
        loop.close()

        # 2. Рендеринг візуальної частини
        tasks[task_id]["progress"] = 45
        raw_video_path = work_dir / "raw_visual.mp4"

        if image_path and os.path.exists(image_path):
            # ----------------------------------------------------
            # Image-to-Video: Ken Burns ефект з кількома сегментами,
            # зум + панорама, кольорокорекція та віньєтка
            # ----------------------------------------------------
            tasks[task_id]["message"] = "Image-to-Video: Ken Burns динамічний зум та панорама..."
            
            # Підготовка вхідного фото під цільовий розмір
            scaled_input = work_dir / "scaled_input.png"
            prep_cmd = [
                "ffmpeg", "-y", "-i", image_path,
                "-vf", f"scale={width*2}:{height*2}:force_original_aspect_ratio=increase,crop={width*2}:{height*2}",
                str(scaled_input)
            ]
            subprocess.run(prep_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            total_frames = duration * fps
            # Ken Burns вираз для zoompan: плавний зум від 1.0 до 1.35 з панорамою
            zoompan_filter = (
                f"zoompan=z='min(zoom+0.0012,1.35)':"
                f"x='iw/4-(iw/zoom/4)*sin(on/{fps}*0.6)':"
                f"y='ih/4+(ih/zoom/4)*cos(on/{fps}*0.5)':"
                f"d={total_frames}:s={width}x{height}:fps={fps},"
                f"eq=contrast=1.08:brightness=0.03:saturation=1.12,"
                f"vignette=PI/4.5"
            )

            cmd_visual = [
                "ffmpeg", "-y", "-loop", "1", "-i", str(scaled_input),
                "-vf", zoompan_filter,
                "-t", str(duration),
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264",
                "-preset", "veryfast",
                str(raw_video_path)
            ]
            subprocess.run(cmd_visual, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        else:
            # ----------------------------------------------------
            # Text-to-Video: Процедурний анімований космічний градієнт
            # з частинками (numpy + PIL)
            # ----------------------------------------------------
            tasks[task_id]["message"] = "Text-to-Video: Процедурний рендеринг часток і градієнта..."
            frames_dir = work_dir / "frames"
            total_frames = duration * fps
            generate_procedural_background_frames(frames_dir, width, height, total_frames, fps, prompt)

            tasks[task_id]["progress"] = 70
            tasks[task_id]["message"] = "Збирання відеопотоку через FFmpeg..."
            cmd_visual = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%04d.png"),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                str(raw_video_path)
            ]
            subprocess.run(cmd_visual, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 3. Накладення Title-картки на перші 2 сек, водяного знаку та аудіо
        tasks[task_id]["progress"] = 80
        tasks[task_id]["message"] = "Додавання водяного знаку та зведення аудіо..."

        title_img_path = work_dir / "title_card.jpg"
        create_title_card(prompt, width, height, title_img_path)

        # Фільтр для водяного знаку "ТЕСТОВИЙ РЕЖИМ (FREE)" та title-overlay на перші 2 секунди
        # Використовуємо drawtext для чіткого напівпрозорого водяного знаку у правому верхньому кутку
        watermark_text = "ТЕСТОВИЙ РЕЖИМ (FREE)"
        vf_complex = (
            f"[0:v]drawtext=text='{watermark_text}':x=w-tw-24:y=24:"
            f"fontsize={max(14, width//45)}:fontcolor=white@0.65:"
            f"box=1:boxcolor=black@0.45:boxborderw=6[v_watermarked];"
            f"[1:v]scale={width}:{height},format=rgba,fade=t=out:st=1.6:d=0.4:alpha=1[v_title];"
            f"[v_watermarked][v_title]overlay=0:0:enable='between(t,0,2)'[v_out]"
        )

        final_cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_video_path),
            "-loop", "1", "-t", "2", "-i", str(title_img_path)
        ]

        if has_voice and audio_path.exists():
            final_cmd.extend([
                "-i", str(audio_path),
                "-filter_complex", vf_complex,
                "-map", "[v_out]",
                "-map", "2:a",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(final_video_path)
            ])
        else:
            # Генерація приємного синтезованого ембієнт-тону якщо голосового запису немає
            ambient_audio_filter = "aevalsrc=sin(220*2*PI*t)*0.08+sin(440*2*PI*t)*0.04+sin(330*2*PI*t)*0.06:s=44100:d=" + str(duration)
            final_cmd.extend([
                "-f", "lavfi", "-i", ambient_audio_filter,
                "-filter_complex", vf_complex,
                "-map", "[v_out]",
                "-map", "2:a",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                "-t", str(duration),
                str(final_video_path)
            ])

        subprocess.run(final_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 4. Генерація мініатюри (thumbnail)
        tasks[task_id]["progress"] = 95
        tasks[task_id]["message"] = "Створення мініатюри..."
        thumb_cmd = [
            "ffmpeg", "-y", "-ss", "00:00:02.5", "-i", str(final_video_path),
            "-vframes", "1", "-q:v", "2", str(final_thumb_path)
        ]
        # Якщо відео дуже коротке або 2.5 сек за межами, робимо з 00:00:01
        res = subprocess.run(thumb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not final_thumb_path.exists() or final_thumb_path.stat().st_size == 0:
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:00.5", "-i", str(final_video_path),
                "-vframes", "1", "-q:v", "2", str(final_thumb_path)
            ], check=True)

        # Збереження в історію
        history_item = {
            "video_id": video_id,
            "title": prompt[:60] + ("..." if len(prompt) > 60 else ""),
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "has_image": bool(image_path),
            "created_at": time.time(),
            "video_url": f"/video/{video_id}",
            "thumbnail_url": f"/thumbnail/{video_id}"
        }
        save_history_item(history_item)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "Відео успішно створено!"
        tasks[task_id]["video_id"] = video_id
        logger.info(f"Завдання {task_id} успішно завершено: {video_id}")

    except Exception as e:
        logger.error(f"Помилка в генерації відео завдання {task_id}: {e}", exc_info=True)
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["progress"] = 0
        tasks[task_id]["message"] = f"Помилка генерації: {str(e)}"
        tasks[task_id]["error"] = str(e)

    finally:
        # Очищення тимчасових файлів
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

# =========================================================================
# ЕНДПОІНТИ REST API
# =========================================================================

@app.get("/status")
def get_server_status():
    """Перевірка стану сервера та системних можливостей"""
    ffmpeg_available = bool(shutil.which("ffmpeg"))
    return {
        "status": "online",
        "app_name": "MyVideo AI Server",
        "version": "1.0.0",
        "mode": "paid_kie_seedance" if KIE_API_KEY else ("paid_fal_seedance" if FAL_API_KEY else "free"),
        "ffmpeg_available": ffmpeg_available,
        "capabilities": {
            "text_to_video": True,
            "image_to_video": True,
            "edge_tts": True,
            "veo_paid": bool(KIE_API_KEY or FAL_API_KEY)  # true, якщо задано KIE_API_KEY або FAL_API_KEY
        },
        "system_time": time.time()
    }

@app.post("/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(..., description="Текстовий опис для генерації відео"),
    aspect_ratio: str = Form("16:9", description="Формат: 16:9, 9:16, 1:1"),
    duration: int = Form(5, description="Тривалість: 5 або 10 секунд"),
    mode: str = Form("free", description="Режим: free або paid"),
    image: Optional[UploadFile] = File(None, description="Фото для Image-to-Video")
):
    """
    Створення нового завдання генерації відео.
    """
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Промпт не може бути порожнім")

    # Якщо користувач спробував платний режим — повертаємо чесну помилку
    if mode == "paid":
        generate_video_paid_stub(prompt)

    if aspect_ratio not in ["16:9", "9:16", "1:1"]:
        aspect_ratio = "16:9"

    if duration not in [5, 10]:
        duration = 5

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "Завдання поставлено в чергу...",
        "video_id": None,
        "error": None,
        "created_at": time.time()
    }

    # Збереження завантаженого зображення якщо є
    image_saved_path = None
    if image and image.filename:
        safe_ext = ".jpg"
        if image.filename.lower().endswith(".png"):
            safe_ext = ".png"
        saved_img_name = f"upload_{task_id}{safe_ext}"
        image_saved_path = str(TEMP_DIR / saved_img_name)
        with open(image_saved_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

    # Запуск фонового рендерингу
    background_tasks.add_task(
        generate_video_task,
        task_id=task_id,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        duration=duration,
        image_path=image_saved_path
    )

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Генерацію розпочато",
        "poll_interval_seconds": 1.5
    }

@app.get("/status/{task_id}")
def get_task_status(task_id: str):
    """Опитування статусу генерації кожні 1-2 секунди"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Завдання не знайдено")
    return tasks[task_id]

@app.get("/video/{video_id}")
def get_video_stream(video_id: str):
    """Потоковий перегляд або завантаження готового MP4 відео"""
    clean_id = video_id.replace(".mp4", "")
    target_path = validate_safe_id(clean_id, VIDEOS_DIR, ".mp4")
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Відеофайл не знайдено")
    return FileResponse(
        path=target_path,
        media_type="video/mp4",
        filename=f"myvideo_{clean_id}.mp4"
    )

@app.get("/thumbnail/{video_id}")
def get_thumbnail(video_id: str):
    """Отримання мініатюри відео"""
    clean_id = video_id.replace(".jpg", "").replace(".png", "").replace(".mp4", "")
    target_path = validate_safe_id(clean_id, THUMBNAILS_DIR, ".jpg")
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Мініатюру не знайдено")
    return FileResponse(path=target_path, media_type="image/jpeg")

@app.get("/uploads/{filename}")
def get_uploaded_image(filename: str):
    """
    Публічна роздача завантаженого користувачем фото — потрібна лише для того,
    щоб зовнішній сервіс fal.ai міг завантажити це фото за посиланням
    (image-to-video). Доступ лише по точному, непередбачуваному імені файлу.
    """
    clean_id, ext = os.path.splitext(filename)
    if ext.lower() not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(status_code=400, detail="Непідтримуване розширення файлу")
    target_path = validate_safe_id(clean_id, TEMP_DIR, ext)
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Файл не знайдено")
    media_type = "image/png" if ext.lower() == ".png" else "image/jpeg"
    return FileResponse(path=target_path, media_type=media_type)

@app.get("/history")
def get_video_history():
    """Список усіх створених відео з метаданими"""
    history = load_history()
    # Фільтруємо лише ті, які фізично існують на диску
    valid_history = []
    for item in history:
        vid = item.get("video_id")
        if vid and (VIDEOS_DIR / f"{vid}.mp4").exists():
            valid_history.append(item)
    return valid_history

@app.delete("/video/{video_id}")
def delete_video(video_id: str):
    """Видалення відео та його мініатюри"""
    clean_id = video_id.replace(".mp4", "")
    video_file = validate_safe_id(clean_id, VIDEOS_DIR, ".mp4")
    thumb_file = validate_safe_id(clean_id, THUMBNAILS_DIR, ".jpg")

    deleted = False
    if video_file.exists():
        video_file.unlink()
        deleted = True
    if thumb_file.exists():
        thumb_file.unlink()

    delete_history_item(clean_id)
    return {"status": "ok", "deleted": deleted, "video_id": clean_id}

@app.post("/edit")
def edit_video(req: EditRequest):
    """
    Редагування існуючого відео (зміна швидкості, ефекти кольорокорекції тощо)
    """
    clean_id = req.video_id.replace(".mp4", "")
    source_video = validate_safe_id(clean_id, VIDEOS_DIR, ".mp4")
    if not source_video.exists():
        raise HTTPException(status_code=404, detail="Вихідне відео не знайдено")

    new_id = f"{clean_id}_edit_{str(uuid.uuid4())[:4]}"
    new_video_path = VIDEOS_DIR / f"{new_id}.mp4"
    new_thumb_path = THUMBNAILS_DIR / f"{new_id}.jpg"

    # Визначаємо FFmpeg фільтр для ефекту
    vf_filter = "null"
    af_filter = "anull"

    if req.effect == "speed_0.5":
        vf_filter = "setpts=2.0*PTS"
        af_filter = "atempo=0.5"
    elif req.effect == "speed_1.5":
        vf_filter = "setpts=0.666*PTS"
        af_filter = "atempo=1.5"
    elif req.effect == "speed_2.0":
        vf_filter = "setpts=0.5*PTS"
        af_filter = "atempo=2.0"
    elif req.effect == "vignette":
        vf_filter = "vignette=PI/3"
    elif req.effect == "vintage":
        vf_filter = "curves=vintage,noise=alls=15:allf=t"
    elif req.effect == "cinema":
        vf_filter = "eq=contrast=1.15:saturation=1.25,vignette=PI/4"
    elif req.effect == "bw":
        vf_filter = "hue=s=0"

    cmd = [
        "ffmpeg", "-y", "-i", str(source_video),
        "-vf", vf_filter,
        "-af", af_filter,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "veryfast",
        str(new_video_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail="Не вдалося застосувати ефект до відео")

    # Створюємо нову мініатюру
    subprocess.run([
        "ffmpeg", "-y", "-ss", "00:00:01", "-i", str(new_video_path),
        "-vframes", "1", "-q:v", "2", str(new_thumb_path)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Зберігаємо в історію
    new_item = {
        "video_id": new_id,
        "title": (req.title or "Відредаговане відео") + f" ({req.effect})",
        "prompt": f"Редагування ({req.effect})",
        "aspect_ratio": "16:9",
        "duration": 5,
        "has_image": False,
        "created_at": time.time(),
        "video_url": f"/video/{new_id}",
        "thumbnail_url": f"/thumbnail/{new_id}"
    }
    save_history_item(new_item)

    return {
        "status": "ok",
        "video_id": new_id,
        "video_url": f"/video/{new_id}",
        "thumbnail_url": f"/thumbnail/{new_id}"
    }

if __name__ == "__main__":
    import uvicorn
    # PORT задає хостинг (напр. Render) через змінну середовища; локально — 8000
    port = int(os.environ.get("PORT", 8000))
    # Запуск на всіх мережевих інтерфейсах для доступу через Wi-Fi, емулятор та хмару
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
