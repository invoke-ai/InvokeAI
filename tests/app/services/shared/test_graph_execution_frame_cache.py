import gc
import weakref

from invokeai.app.services.shared.graph import Graph, GraphExecutionState


def test_execution_frames_are_cached_per_state_without_a_4096_entry_cutoff() -> None:
    state = GraphExecutionState(graph=Graph())
    runtime_state = state.__pydantic_private__["_runtime_state"]
    first_frame = state._engine_frame((0,))
    first_frame_ref = weakref.ref(first_frame)

    for index in range(1, 4_200):
        state._engine_frame((index,))

    assert len(runtime_state.engine_frames) == 4_200
    assert state._engine_frame((0,)) is first_frame

    same_state_copy = state.model_copy()
    assert same_state_copy._engine_frame((0,)) is first_frame
    assert same_state_copy.__pydantic_private__["_runtime_state"].engine_frames is not runtime_state.engine_frames

    new_state_copy = state.model_copy(update={"id": "different-state"})
    assert new_state_copy._engine_frame((0,)) is not first_frame
    assert len(new_state_copy.__pydantic_private__["_runtime_state"].engine_frames) == 1

    del first_frame
    del same_state_copy
    del new_state_copy
    del runtime_state
    del state
    gc.collect()

    assert first_frame_ref() is None
