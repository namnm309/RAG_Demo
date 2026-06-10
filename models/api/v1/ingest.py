from typing import List

from pydantic import BaseModel, Field


class IngestFileResultData(BaseModel):
    file_name: str
    success: bool
    chunks_created: int = 0
    message: str = ""
    validation_errors: List[str] = Field(default_factory=list)


class IngestData(BaseModel):
    message: str
    documents_loaded: int = 0
    chunks_created: int = 0
    files: List[str] = Field(default_factory=list)
    failed_files: List[str] = Field(default_factory=list)
    file_results: List[IngestFileResultData] = Field(default_factory=list)
