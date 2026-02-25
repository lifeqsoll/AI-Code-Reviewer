import typer
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer()
console = Console()

@app.command()
def main(path: str):
    """
    Анализ кода через AI. Путь к файлу или URL репозитория.
    """
    console.print(f"[bold blue]🚀 Анализ запущен для: {path}[/bold blue]")
    console.print("[yellow]Это может занять несколько минут для больших файлов...[/yellow]")
    
    try:
        # Используем прогресс-бар, чтобы пользователь видел, что процесс идет
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Ожидание ответа от сервера...", total=None)
            
            response = requests.post(
                "http://localhost:8000/review", 
                json={"source": path},
                timeout=1200  # 20 минут для очень больших задач
            )
        
        response.raise_for_status()
        report_md = response.json().get("report", "Ошибка: Отчет пуст.")
        
        console.print("\n[bold green]✅ Отчет готов:[/bold green]\n")
        console.print(Markdown(report_md))
        
    except requests.exceptions.Timeout:
        console.print("[red]Ошибка: Превышено время ожидания сервера (Timeout). Код слишком большой.[/red]")
    except Exception as e:
        console.print(f"[red]Ошибка связи с сервером: {e}[/red]")

if __name__ == "__main__":
    app()
