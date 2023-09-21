from pydantic import BaseModel, Field


class InvocationCacheStatus(BaseModel):
    size: int = Field(description="The current size of the invocation cache")
    hits: int = Field(description="The number of cache hits")
    misses: int = Field(description="The number of cache misses")
    enabled: bool = Field(description="Whether the invocation cache is enabled")
    max_size: int = Field(description="The maximum size of the invocation cache")
