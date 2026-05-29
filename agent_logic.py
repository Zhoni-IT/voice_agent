import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

load_dotenv()

# --- 1. ИНСТРУМЕНТЫ (TOOLS) ---

@tool
def check_balance() -> str:
    """Используй этот инструмент, чтобы проверить баланс топлива или денег на аккаунте пользователя."""
    return "На вашем балансе 45 литров АИ-95 и 2500 тенге."

@tool
def create_order(fuel_type: str, liters: int) -> str:
    """Используй этот инструмент, чтобы создать заказ на заправку. Передай тип топлива и количество литров."""
    return f"Заказ успешно создан! Заправка {liters} литров {fuel_type} активирована на колонке номер 3."

@tool
def get_history() -> str:
    """Используй этот инструмент, чтобы посмотреть историю последних заправок пользователя."""
    return "Последняя заправка: 15 мая, 30 литров АИ-95 на сумму 7500 тенге."

@tool
def change_limits(limit_type: str, amount: int) -> str:
    """Используй этот инструмент, чтобы изменить дневной или разовый лимит на заправку (в литрах или тенге).
    Например, лимит на количество литров или сумму денег.
    """
    return f"Лимит '{limit_type}' успешно изменен на новое значение: {amount}."

# Регистрируем ВСЕ 4 инструмента
tools = [check_balance, create_order, get_history, change_limits]
tools_map = {t.name: t for t in tools}

# --- 2. НАСТРОЙКА МОДЕЛИ И ПРОМПТА ---

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)

system_message = SystemMessage(content="""Ты — вежливый голосовой AI-ассистент цифровой заправки "Смарт Заправка".
Твоя цель — помогать водителям заправлять машину, проверять баланс, смотреть историю и менять лимиты.

ПРАВИЛА ПОВЕДЕНИЯ:
1. Отвечай ОЧЕНЬ кратко и емко (1-2 предложения), так как твои ответы будут озвучиваться голосом.
2. Не используй списки, дефисы, звездочки (*) или решетки (#). Пиши все только понятными словами.
3. ОБРАБОТКА ВНЕШНИХ ТЕМ: Если пользователь спрашивает о вещах, не связанных с заправкой, топливом или его аккаунтом (например, про погоду, политику, программирование), вежливо ответь: "Я могу помочь вам только с вопросами нашей цифровой заправки. Хотите проверить баланс или заказать топливо?".
4. ОБРАБОТКА НЕПОНЯТНЫХ КОМАНД: Если фраза пользователя обрывочна или непонятна, ответь: "Извините, я не совсем вас понял. Повторите, пожалуйста, ваш запрос".""")

memory_history = []

# --- 3. ЦИКЛ АГЕНТА (AGENT LOOP) ---

def ask_agent(user_input: str) -> str:
    global memory_history
    
    # Строим цепочку: системный промпт + история + новый вопрос
    messages = [system_message] + memory_history + [HumanMessage(content=user_input)]
    
    # Первый вызов — с инструментами (модель решает, вызывать ли функцию)
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        # Если ИИ выбрал вызвать инструмент(ы), добавляем его решение в общую цепочку
        messages.append(response)
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\n🛠️ [Agent] Вызов функции: {tool_name} с параметрами: {tool_args}")
            
            if tool_name in tools_map:
                tool_output = tools_map[tool_name].invoke(tool_args)
            else:
                tool_output = f"Ошибка: Инструмент {tool_name} не найден."
                
            print(f"📝 [Agent] Результат функции: {tool_output}")
            # Добавляем результат работы функции в цепочку
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
        
        # === ВОТ ЗДЕСЬ ИСПРАВЛЕНИЕ ===
        # Используем ЧИСТЫЙ llm вместо llm_with_tools.
        # Это убирает баг Groq и гарантирует, что ИИ вернет текст, а не пустую строку!
        final_response = llm.invoke(messages)
        
        # Сохраняем чистый диалог в память (без технических логов вызова функций)
        memory_history.append(HumanMessage(content=user_input))
        memory_history.append(final_response)
        
        return final_response.content
    else:
        # Если инструменты не понадобились, просто сохраняем обычный текстовый ответ
        memory_history.append(HumanMessage(content=user_input))
        memory_history.append(response)
        return response.content