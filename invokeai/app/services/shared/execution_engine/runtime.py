"""State-local registries for generic execution primitives.

The registries are deliberately independent of ``GraphExecutionState``.  The
state owns the compatibility adapter and persists the resulting records using
its existing runtime snapshot fields; this module owns identity and lifecycle
invariants.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from invokeai.app.services.shared.execution_engine.primitives import (
    ActivationGate,
    ContinuationRecord,
    ExecutionFrame,
    StreamBuffer,
)


class ExecutionEngineRuntime(BaseModel):
    """Typed, frame-scoped runtime records used by compatibility adapters."""

    model_config = ConfigDict(extra="forbid")

    gates: dict[str, ActivationGate] = Field(default_factory=dict)
    streams: dict[str, StreamBuffer[Any]] = Field(default_factory=dict)
    continuations: dict[str, ContinuationRecord[Any]] = Field(default_factory=dict)

    def register_gate(
        self,
        gate_id: str,
        owner_id: str,
        frame: ExecutionFrame,
        branches: tuple[str, ...],
    ) -> ActivationGate:
        existing = self.gates.get(gate_id)
        if existing is not None:
            if existing.owner_id != owner_id or existing.frame != frame or existing.branches != branches:
                raise ValueError(f"activation gate identity conflict: {gate_id}")
            return existing
        gate = ActivationGate(gate_id=gate_id, owner_id=owner_id, frame=frame, branches=branches)
        self.gates[gate_id] = gate
        return gate

    def get_gate(self, gate_id: str) -> ActivationGate:
        try:
            return self.gates[gate_id]
        except KeyError as exc:
            raise KeyError(f"activation gate not registered: {gate_id}") from exc

    def replace_gate(self, gate: ActivationGate) -> None:
        self.gates[gate.gate_id] = gate

    def remove_gate(self, gate_id: str) -> None:
        self.gates.pop(gate_id, None)

    def resolve_gate(self, gate_id: str, owner_id: str, frame: ExecutionFrame, branch: str) -> bool:
        gate = self.get_gate(gate_id)
        gate.assert_scope(owner_id, frame)
        return gate.resolve(branch)

    def get_or_create_stream(self, stream_id: str, owner_id: str, frame: ExecutionFrame) -> StreamBuffer[Any]:
        existing = self.streams.get(stream_id)
        if existing is not None:
            existing.assert_scope(owner_id, frame)
            return existing
        stream = StreamBuffer[Any](stream_id=stream_id, owner_id=owner_id, frame=frame)
        self.streams[stream_id] = stream
        return stream

    def get_stream(self, stream_id: str) -> StreamBuffer[Any]:
        try:
            return self.streams[stream_id]
        except KeyError as exc:
            raise KeyError(f"stream not registered: {stream_id}") from exc

    def replace_stream(self, stream: StreamBuffer[Any]) -> None:
        self.streams[stream.stream_id] = stream

    def remove_stream(self, stream_id: str) -> None:
        self.streams.pop(stream_id, None)

    def register_continuation(
        self, continuation_id: str, owner_id: str, frame: ExecutionFrame, kind: str
    ) -> ContinuationRecord[Any]:
        existing = self.continuations.get(continuation_id)
        if existing is not None:
            existing.assert_scope(owner_id, frame)
            if existing.kind != kind:
                raise ValueError(f"continuation identity conflict: {continuation_id}")
            return existing
        continuation = ContinuationRecord[Any](
            continuation_id=continuation_id,
            owner_id=owner_id,
            frame=frame,
            kind=kind,
        )
        self.continuations[continuation_id] = continuation
        return continuation

    def replace_continuation(self, continuation: ContinuationRecord[Any]) -> None:
        self.continuations[continuation.continuation_id] = continuation

    def remove_continuation(self, continuation_id: str) -> None:
        self.continuations.pop(continuation_id, None)

    def get_continuation(self, continuation_id: str) -> ContinuationRecord[Any]:
        try:
            return self.continuations[continuation_id]
        except KeyError as exc:
            raise KeyError(f"continuation not registered: {continuation_id}") from exc
