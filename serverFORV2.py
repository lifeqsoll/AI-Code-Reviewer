import ast
import os
import shutil
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from git import Repo

# Настройка логирования для отслеживания прогресса в терминале
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Инициализация модели DeepSeek-Coder-V2 Lite
# Параметры подобраны для 8 ГБ VRAM: модель ~4.5ГБ + контекст ~2ГБ
llm = OllamaLLM(
    model="deepseek-coder-v2:16b-lite-instruct-q4_K_M", 
    temperature=0.1,
    num_ctx=4192,     # Размер контекста (4к токенов) для анализа длинных файлов(с 8 к оч долго)
    num_gpu=99,       # Принудительная загрузка всех слоев в видеокарту
    repeat_penalty=1.1
)

class ReviewRequest(BaseModel):
    source: str

def get_code_chunks(code: str, file_path: str, max_lines: int = 400):
    """
    Разбивает код на логические блоки. 
    max_lines увеличен до 400, так как у V2 контекст больше.
    """
    try:
        tree = ast.parse(code)
        chunks = []
        lines = code.splitlines()
        
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start, end = node.lineno - 1, node.end_lineno
                content = "\n".join(lines[start:end])
                
                # Если блок все равно слишком велик для одного прохода
                if (end - start) > max_lines:
                    for i in range(0, len(lines[start:end]), max_lines):
                        sub_content = "\n".join(lines[start:end][i:i + max_lines])
                        chunks.append({
                            "name": f"{getattr(node, 'name', 'Block')} (часть {i//max_lines + 1})", 
                            "content": sub_content
                        })
                else:
                    chunks.append({"name": getattr(node, 'name', 'Block'), "content": content})
        
        if not chunks:
            # Если файл не имеет структуры (простой скрипт), режем по строкам
            for i in range(0, len(lines), max_lines):
                chunks.append({"name": f"Часть {i//max_lines + 1}", "content": "\n".join(lines[i:i + max_lines])})
                
        return chunks
    except Exception as e:
        logger.error(f"Ошибка парсинга AST в {file_path}: {e}")
        lines = code.splitlines()
        return [{"name": "Raw Chunk", "content": "\n".join(lines[i:i+max_lines])} for i in range(0, len(lines), max_lines)]

SYSTEM_PROMPT = """
Ты — Senior Python Developer и эксперт по безопасности. 
Проведи глубокий аудит кода блока: {name}.
Сфокусируйся на критических ошибках, утечках памяти и безопасности.

Формат ответа Markdown:
### 🐞 Критические ошибки и баги
### ⚡ Оптимизация производительности
### 🛡 Безопасность и уязвимости
### 📊 Сложность и читаемость
"""

@app.post("/review")
def review(req: ReviewRequest):
    work_dir = "temp_review"
    
    # Очистка и подготовка файлов
    if req.source.startswith("http"):
        if os.path.exists(work_dir): 
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.info("Старая временная папка удалена.")
            
        logger.info(f"Клонирование репозитория: {req.source}")
        try:
            Repo.clone_from(req.source, work_dir, depth=1)
            files = []
            for dp, dn, filenames in os.walk(work_dir):
                if ".git" in dp: continue # Пропускаем системные файлы гита
                for f in filenames:
                    if f.endswith('.py'):
                        files.append(os.path.join(dp, f))
        except Exception as e:
            logger.error(f"Ошибка Git: {e}")
            return {"report": f"Ошибка клонирования: {e}"}
    else:
        # Если передан путь к локальному файлу
        files = [req.source] if os.path.isfile(req.source) else []

    if not files:
        logger.warning("Файлы для анализа не обнаружены.")
        return {"report": "Python файлы не найдены по указанному пути."}

    logger.info(f"Найдено файлов: {len(files)}. Начинаю анализ...")
    full_report = []

    for file_path in files:
        logger.info(f">>> Анализ файла: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            chunks = get_code_chunks(content, file_path)
            file_results = []
            
            for chunk in chunks:
                logger.info(f"    Отправка в LLM: {chunk['name']}")
                prompt = PromptTemplate.from_template(SYSTEM_PROMPT).format(name=f"{file_path} -> {chunk['name']}")
                
                try:
                    # Вызов нейросети
                    res = llm.invoke(f"{prompt}\n\nКод:\n{chunk['content']}")
                    file_results.append(f"#### Блок: {chunk['name']}\n{res}")
                except Exception as e:
                    logger.error(f"Ошибка при вызове LLM: {e}")
            
            full_report.append(f"## Файл: {file_path}\n" + "\n".join(file_results))
            
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {e}")

    logger.info("Весь репозиторий успешно проанализирован.")
    return {"report": "\n\n---\n\n".join(full_report)}

if __name__ == "__main__":
    import uvicorn
    # Запуск сервера
    uvicorn.run(app, host="0.0.0.0", port=8000)

