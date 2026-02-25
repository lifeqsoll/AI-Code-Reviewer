import ast
import os
import shutil
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from git import Repo

# Настройка логирования, чтобы видеть процесс в терминале
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
llm = OllamaLLM(model="deepseek-coder", temperature=0.1)

class ReviewRequest(BaseModel):
    source: str

def get_code_chunks(code: str, file_path: str, max_lines: int = 150):
    """Разбивает код на части. Если блок слишком большой, режет его по строкам."""
    try:
        tree = ast.parse(code)
        chunks = []
        lines = code.splitlines()
        
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start, end = node.lineno - 1, node.end_lineno
                content = "\n".join(lines[start:end])
                
                # Если блок слишком большой, делим его на под-части
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
            # Если ast не нашел структур (простой скрипт), режем просто по строкам
            for i in range(0, len(lines), max_lines):
                chunks.append({"name": f"Chunk {i//max_lines + 1}", "content": "\n".join(lines[i:i + max_lines])})
                
        return chunks
    except Exception as e:
        logger.error(f"Ошибка парсинга {file_path}: {e}")
        # В случае ошибки парсинга (например, синтаксис), режем файл просто по строкам
        lines = code.splitlines()
        return [{"name": "Raw Chunk", "content": "\n".join(lines[i:i+max_lines])} for i in range(0, len(lines), max_lines)]

SYSTEM_PROMPT = """
Ты — Senior Developer. Проведи аудит кода блока {name}.
Пиши кратко и по делу.
Формат Markdown:
### 🐞 Баги
### ⚡ Оптимизация
### 📊 Сложность
"""

@app.post("/review")
def review(req: ReviewRequest):
    work_dir = "temp_review"
    
    # 1. Подготовка файлов
    if req.source.startswith("http"):
        if os.path.exists(work_dir): 
            shutil.rmtree(work_dir, ignore_errors=True)
        logger.info(f"Клонирование репозитория: {req.source}")
        try:
            Repo.clone_from(req.source, work_dir, depth=1)
        except Exception as e:
            return {"report": f"Ошибка клонирования: {e}"}
        
        files = []
        for dp, dn, filenames in os.walk(work_dir):
            for f in filenames:
                if f.endswith('.py'):
                    files.append(os.path.join(dp, f))
    else:
        files = [req.source] if os.path.isfile(req.source) else []

    if not files:
        return {"report": "Python файлы не найдены."}

    logger.info(f"Найдено файлов для анализа: {len(files)}")
    full_report = []

    # 2. Цикл анализа
    for file_path in files:
        logger.info(f"--- Обработка файла: {file_path} ---")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            chunks = get_code_chunks(content, file_path)
            file_results = []
            
            for chunk in chunks:
                logger.info(f"Анализ блока: {chunk['name']}")
                prompt = PromptTemplate.from_template(SYSTEM_PROMPT).format(name=f"{file_path} -> {chunk['name']}")
                
                # Вызов LLM (это может занять время)
                try:
                    res = llm.invoke(f"{prompt}\n\nКод:\n{chunk['content']}")
                    file_results.append(f"#### Блок: {chunk['name']}\n{res}")
                except Exception as e:
                    logger.error(f"Ошибка LLM на блоке {chunk['name']}: {e}")
            
            full_report.append(f"## Файл: {file_path}\n" + "\n".join(file_results))
            
        except Exception as e:
            logger.error(f"Не удалось прочитать файл {file_path}: {e}")

    logger.info("Анализ полностью завершен.")
    return {"report": "\n\n---\n\n".join(full_report)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
