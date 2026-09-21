from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv(override=True)
from agents.file_summarization_agent import file_summarization_agent
from agents.repo_overview_agent import repo_overview_agent
from agents.file_selection_agent import file_selection_agent
from agents.arch_overview_agent import arch_overview_agent
import asyncio
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

class CodebaseSummarizer:
    INCLUDE_EXTENSIONS = {
        ".py", 
        ".js", ".jsx", ".ts", ".tsx", 
        ".md", ".txt", 
        ".json", ".yaml", ".yml", ".toml",
        ".sh", ".css", ".html",
    }
    INCLUDE_FILENAMES = {"Dockerfile", "Makefile"} 

    def __init__(self):
        pass

    def should_include_file(self, file_path: Path) -> bool:
        if (not file_path.is_file() or file_path.name.startswith('.')): 
            return False
        if (file_path.suffix.lower() in self.INCLUDE_EXTENSIONS or file_path.name in self.INCLUDE_FILENAMES):
            return True
        return False
    
    async def summarize_files(self, dir: Path) -> Dict[str, str]:
        file_summaries: Dict[str, str] = {}
        files_to_process = list(dir.rglob("*"))

        for file_path in tqdm(files_to_process, desc="🔍 Summarizing files", unit="file"):
            if (not self.should_include_file(file_path) or file_path.stat().st_size == 0):
                continue

            text = file_path.read_text(encoding="utf-8")

            user_prompt = """
                Please summarize the following file: `{file_path}`.

                The summary should be a **concise paragraph** (around 40-60 words) that 
                explains the file's primary purpose, its main functions or classes, and how 
                it fits into the broader project. Focus on the *what* and *why*, not a 
                line-by-line explanation of the *how*.

                <file_content>
                {file_content}
                </file_content>
            """

            result = await file_summarization_agent.run(user_prompt.format(file_path=file_path.name, file_content=text))
            summary = result.output
            if summary:
                file_summaries[file_path.name] = summary

        print(f"✅ Summarized {len(file_summaries)} files.")
        return file_summaries

    async def build_repo_overview(self, file_summaries: Dict[str, str]) -> str:
        bullets = "\n".join(f"- **{n}**: {s}" for n, s in file_summaries.items())
        user_prompt = """
            Below is a list of source files with their summaries.

            1. Write an **'Overview'** section (≈3-4 sentences) explaining the purpose of the repository.
            2. Follow it with a **'Key Components'** bullet list (max 6 bullets) referencing the files.
            3. Close with a short 'Getting Started' hint: `pip install -r requirements.txt` etc.

            ---
            FILE SUMMARIES
            {bullets}
        """

        result = await repo_overview_agent.run(user_prompt.format(bullets=bullets))
        return result.output
    
    async def select_important_files(self, file_summaries: Dict[str, str]) -> List[str]:
        bullets = "\n".join(f"- **{n}**: {s}" for n, s in file_summaries.items())
        user_prompt = """
            Based on the following file summaries, identify the most architecturally
            significant files. These files should represent the core logic,
            primary entry points, or key data structures of the project.
            ---
            FILE SUMMARIES
            {bullets}
        """

        result = await file_selection_agent.run(user_prompt.format(bullets=bullets))
        return result.output.important_files
    
    def get_code_snippets(self, dir: Path, important_files: List[str]) -> List[Tuple[str, str]]:
        code_snippets = []
        for fname in important_files:
            file_path = dir / fname
            if file_path.is_file():
                try:
                    code = file_path.read_text(encoding="utf-8")
                    code_snippets.append((fname, code))
                except Exception as e:
                    logfire.error(f"Failed to read {file_path}: {e}")
        return code_snippets
    
    def build_arch_overview_prompt(self, file_summaries: Dict[str, str], code_snippets: List[Tuple[str, str]]) -> str:
        summary_lines = "\n".join(f"- **{n}**: {s}" for n, s in file_summaries.items())
        prompt_sections = [
            "[[FILE_SUMMARIES]]",
            summary_lines,
            "[[/FILE_SUMMARIES]]",
        ]

        if code_snippets:
            code_block_lines = []
            for fname, code in code_snippets:
                added = "\n### " + fname + "\n```code\n" + code + "\n```\n"
                code_block_lines.append(added)
            if code_block_lines:
                prompt_sections.extend(
                    ["[[RAW_CODE_SNIPPETS]]"] + code_block_lines + 
                    ["[[/RAW_CODE_SNIPPETS]]"]
                )
        user_prompt = "\n".join(prompt_sections)
        user_prompt += """
             ---
        **Your tasks**
        1. Identify the major abstractions (classes, services, data models) 
           across the entire codebase.
        2. Explain how they interact – include dependencies, data flow, and any 
           cross-cutting concerns.
        3. Output a concise *Architecture & Key Concepts* section suitable for a 
           README, consisting of:
           • short Overview (≤ 3 sentences)
           • Mermaid diagram (`classDiagram` or `flowchart`) of components
           • bullet list of abstractions with brief descriptions.
        """
        return user_prompt
    
    async def build_arch_overview(self, file_summaries: Dict[str, str], code_snippets: List[Tuple[str, str]]) -> str:
        user_prompt = self.build_arch_overview_prompt(file_summaries, code_snippets) 
        result = await arch_overview_agent.run(user_prompt)
        return result.output  

    def generate_readme_content(self, out_dir: Path, file_summaries: Dict[str, str], repo_overview: str, arch_overview: str) -> str:
        out_dir.mkdir(parents=True, exist_ok=True)
        readme_path = out_dir / f"README.md"
        print(f"\n✍️ Writing final README to {readme_path.resolve()}...")
        with readme_path.open("w", encoding="utf-8") as fh:
            fh.write(f"# Repository Summary\n\n{repo_overview}\n\n")
            fh.write("## Architecture & Key Concepts\n\n")
            fh.write(f"{arch_overview}\n\n")
            fh.write("## File Summaries\n\n")
            for n, s in sorted(file_summaries.items()):
                fh.write(f"- **{n}** – {s}\n")
        print(f"\n\n🎉 Success! Documentation generated at: {readme_path.resolve()}") 

async def summarize(in_dir: Path, out_dir: Path):
    summarizer = CodebaseSummarizer()
    file_summaries = await summarizer.summarize_files(in_dir)
    repo_overview = await summarizer.build_repo_overview(file_summaries)
    important_files = await summarizer.select_important_files(file_summaries)
    print(f"📌 Important files identified: {important_files}")
    code_snippets = summarizer.get_code_snippets(in_dir, important_files)
    arch_overview = await summarizer.build_arch_overview(file_summaries, code_snippets)
    summarizer.generate_readme_content(out_dir, file_summaries, repo_overview, arch_overview)

if __name__ == "__main__":
    base = Path(__file__).parent / "data"
    in_dir = base / "repo"
    out_dir = base / "output"
    print(in_dir)
    asyncio.run(summarize(in_dir, out_dir))