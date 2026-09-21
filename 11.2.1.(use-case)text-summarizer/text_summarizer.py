from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
from agents.mapper_agent import mapper_agent
from agents.reducer_agent import reducer_agent
from agents.summarizer_agent import summarizer_agent
from agents.embedder import embedder
from config.config_reader import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
import logfire
from sklearn.cluster import KMeans
import numpy as np

logfire.configure()
logfire.instrument_pydantic_ai()

class TextSummarizer:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)

    async def summarize_short_docs(self, text: str) -> str:
        res = await summarizer_agent.run(text)
        return res.output

    def chunk_text(self, text: str, chunk_size: int = settings.CHUNK_SIZE) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.1), separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )
        return text_splitter.split_text(text)
    
    async def summarize_medium_docs(self, text: str) -> str:
        chunks = self.chunk_text(text)

        mapper_user_prompt ="""
            Your role is to create a concise, factual summary of a text chunk from the document.
            - Extract only key facts, figures, and statements from the chunk text itself.
            - Omit any conversational introductions or conclusions. Do not explain what you 
                are doing.
            - If a chunk contains no substantive information (e.g., only headers, formatting, 
                or boilerplate), output the exact phrase: "No substantive information."

            **Text Chunk:**
            {chunk_text}
        """

        async def wrapped_mapper(chunk):
            async with self.semaphore:
                result = await mapper_agent.run(mapper_user_prompt.format(chunk_text=chunk))
                return result.output
        
        tasks = [wrapped_mapper(chunk) for chunk in chunks]
        chunk_summaries = await asyncio.gather(*tasks)

        reducer_user_prompt = """
            You are a research assistant tasked with creating an executive summary.
            You have been given a series of concise summaries from different sections of a document.
            Your goal is to synthesize these individual summaries into a single, well-written, 
            and coherent executive summary.
            The final summary should read like a standalone document, flowing logically from 
            one topic to the next.

            **Summaries of Report Sections:**
            {chunk_summaries}
        """
        summaries_text  = "\\n\\n---\\n\\n".join(chunk_summaries)
        res = await reducer_agent.run(reducer_user_prompt.format(chunk_summaries=summaries_text))
        return res.output

    async def summarize_long_docs(self, text: str) -> str:
        chunks = self.chunk_text(text)
        vectors = await embedder.embed_documents(chunks)

        nclusters = 5
        model = KMeans(n_init=10, n_clusters=nclusters, random_state=0).fit(vectors)

        closest_indices = []
        for i in range(nclusters):
            distances = np.linalg.norm(vectors - model.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)
        selected_indices = sorted(closest_indices)
        selected_chunks = [chunks[idx] for idx in selected_indices]

        return await self.summarize_medium_docs(chunks=selected_chunks)

    async def summarize(self, text: str) -> str:
        if len(text) <= settings.SHORT_DOC_THRESHOLD:
            return await self.summarize_short_docs(text)
        elif len(text) <= settings.MEDIUM_DOC_THRESHOLD:
            return await self.summarize_medium_docs(text)
        elif len(text) <= settings.LONG_DOC_THRESHOLD:
            return await self.summarize_long_docs(text)
        else:
            return "Not Supported"

if __name__ == "__main__":
    file = Path(__file__).parent / "data/exam-project.md"
    print(file)
    text = file.read_text(encoding="utf-8")
    print(f"Original Text Length: {len(text)}")

    summarizer = TextSummarizer()
    summary = asyncio.run(summarizer.summarize(text))
    print(f"Summary: {summary}")
