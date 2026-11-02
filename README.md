# MyVideo AI - Сервер локальної генерації відео

Бекенд на FastAPI + FFmpeg + Edge-TTS для додатку MyVideo AI.

## Швидкий запуск на ПК користувача

### 1. Вимоги:
- Python 3.10 або новіший
- Встановлений FFmpeg у системі:
  - **Windows**: `winget install Gyan.FFmpeg` або завантажте з ffmpeg.org і додайте в PATH
  - **macOS**: `brew install ffmpeg`
  - **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install -y ffmpeg`

### 2. Встановлення залежностей:
```bash
cd server
pip install -r requirements.txt
```

### 3. Запуск сервера:
```bash
python main.py
```
або через uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Підключення з додатку Android:
- **У Android Studio Емуляторі:** використовуйте адресу за замовчуванням `http://10.0.2.2:8000`
- **На реальному смартфоні через спільний Wi-Fi:** дізнайтеся локальну IP-адресу комп'ютера (наприклад `192.168.1.150`) та вкажіть у налаштуваннях додатку `http://192.168.1.150:8000`.

## Ендпоінти:
- `GET /status` — статус сервера, перевірка з'єднання та наявності FFmpeg
- `POST /generate` — створення відео (текстовий опис або фото + Ken Burns, озвучення uk-UA)
- `GET /status/{task_id}` — опитування статусу рендерингу (відсоток готовності)
- `GET /video/{video_id}` — потокове відтворення/завантаження готового MP4
- `GET /thumbnail/{video_id}` — мініатюра відео
- `POST /edit` — редагування відео (швидкість, фільтри)
- `DELETE /video/{video_id}` — видалення відео
- `GET /history` — історія згенерованих відео
