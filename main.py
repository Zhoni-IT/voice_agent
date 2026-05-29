import os
import urllib.parse
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import edge_tts
from dotenv import load_dotenv

# Импортируем логику нашего агента
from agent_logic import ask_agent

load_dotenv()

app = FastAPI(
    title="Smart Filling Station Voice Agent API",
    description="Бэкенд для голосового ассистента заправки (FastAPI + LangChain + Groq + Edge-TTS)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Бэкенд голосового ассистента заправки активен!"}

# --- 1. ТЕКСТОВЫЙ РЕЖИМ ---
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст запроса не может быть пустым")
    
    try:
        agent_reply = ask_agent(request.text)
        return {"status": "success", "user_input": request.text, "agent_reply": agent_reply}
    except Exception as e:
        print(f"❌ Ошибка в /chat: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка работы агента: {str(e)}")

# --- 2. ПОЛНОЦЕННЫЙ ГОЛОСОВОЙ РЕЖИМ (Voice-to-Voice) ---
@app.post("/voice-input")
async def voice_input_endpoint(file: UploadFile = File(...)):
    """Принимает аудиозапись, обрабатывает агентом и возвращает готовый аудио-ответ (.mp3)"""
    try:
        file_suffix = os.path.splitext(file.filename)[1] or ".wav"
        audio_bytes = await file.read()
        
        # Создаем временный файл для входящего аудио
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        print(f"🎙️ [API] Получен аудиофайл {file.filename}, отправка в Groq Whisper STT...")

        # Шаг 1: Распознавание речи (STT)
        with open(temp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(file.filename, audio_file.read()),
                model="whisper-large-v3",
                language="ru",
                response_format="json"
            )
        
        os.unlink(temp_path)  # Удаляем временный файл входящего аудио
        user_text = transcription.text
        print(f"🗣️ [API] Покупатель сказал: '{user_text}'")
        
        if not user_text.strip():
            agent_reply = "Извините, я не услышал вашей команды. Повторите, пожалуйста."
        else:
            # Шаг 2: Обработка текста нашим Лэнгчейн-агентом (и вызов инструментов)
            agent_reply = ask_agent(user_text)
        
        print(f"🤖 [API] Ответ агента: '{agent_reply}'")

        # Шаг 3: Синтез речи (TTS) через Edge-TTS
        # Используем отличный мужской голос 'ru-RU-DmitryNeural' (или 'ru-RU-SvetlanaNeural' для женского)
        tts = edge_tts.Communicate(agent_reply, "ru-RU-DmitryNeural")
        
        # Создаем временный файл для исходящего ответа
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        output_path = temp_output.name
        temp_output.close()  # Закрываем дескриптор, чтобы edge-tts мог записать файл
        
        await tts.save(output_path)
        print(f"🔊 [API] Аудио-ответ успешно сгенерирован.")

        # Шаг 4: Отправляем аудиофайл обратно пользователю
        # Используем urllib.parse.quote, чтобы безопасно закодировать кириллицу для HTTP-заголовков
        return FileResponse(
            path=output_path,
            media_type="audio/mpeg",
            filename="agent_response.mp3",
            headers={
                "X-User-Text": urllib.parse.quote(user_text), 
                "X-Agent-Reply": urllib.parse.quote(agent_reply)
            }
        )

    except Exception as e:
        print(f"❌ Ошибка в /voice-input: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки голосового ввода: {str(e)}")