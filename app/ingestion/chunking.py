from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.models import Chunk, Page


def chunk_pages(
    pages: list[Page], document_id: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Split each page into overlapping chunks that remember their page number.

    The splitter tries paragraph breaks first, then lines, then sentences, then
    words, so chunks end at natural boundaries where possible. Chunks never span
    pages, which keeps every citation pointing at exactly one page.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for page in pages:
        for text in splitter.split_text(page.text):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-c{index}",
                    document_id=document_id,
                    page=page.number,
                    index=index,
                    text=text,
                )
            )
    return chunks
