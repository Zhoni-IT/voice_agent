import os
import time
from dotenv import load_dotenv
from groq import Groq
import speech_recognition as sr
from gtts import gTTS

# Загружаем переменные окружения из файла .env
load_dotenv()

# Инициализируем клиента Groq
groq_client = Groq()

def listen_and_transcribe():
    """
    Записывает голос с микрофона, сохраняет в WAV и отправляет в Groq Whisper API.
    """
    recognizer = sr.Recognizer()
    temp_wav = "request.wav"
    
    with sr.Microphone() as source:
        print("\n🤖 Я тебя слушаю... Говори!")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio_data = recognizer.listen(source)
        
    print("🤖 Обрабатываю аудио через Groq Whisper...")
    
    with open(temp_wav, "wb") as f:
        f.write(audio_data.get_wav_data())
        
    try:
        with open(temp_wav, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language="ru",
                response_format="text"
            )
        
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        print(f"Ты сказал: {transcription.strip()}")
        return transcription.strip()
        
    except Exception as e:
        print(f"🤖 Ошибка распознавания: {e}")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return None


def speak(text):
    """
    Превращает текст в речь и гарантированно воспроизводит на Windows.
    """
    print(f"Робот отвечает: {text}")
    
    tts = gTTS(text=text, lang='ru')
    filename = "reply.mp3"
    tts.save(filename)
    
    # ФУЛПРУФ-МЕТОД ДЛЯ WINDOWS: запускаем файл через стандартное приложение
    # Это откроет ваш плеер по умолчанию (Groove Music, Windows Media Player и т.д.)
    os.system(f"start {filename}")
    
    # Даем плееру 4 секунды, чтобы проиграть звук, прежде чем продолжить
    time.sleep(4)


def main():
    speak("Привет! Проверка звука. Если ты меня слышишь, значит всё работает!")
    
    user_text = listen_and_transcribe()
    
    if user_text and len(user_text) > 0:
        reply = f"Отлично! Я услышал фразу: {user_text}"
        speak(reply)
    else:
        speak("Я ничего не услышал, попробуй еще раз.")


if __name__ == "__main__":
    main()