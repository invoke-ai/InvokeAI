import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { assert } from 'tsafe';

export const buildSaveVideoGraph = ({
    state,
}: {
    state: RootState;
}): { graph: Graph; outputNodeId: string } => {

    const taskId = state.video.generatedVideo?.taskId;

    assert(taskId, 'No task ID found in state');

    const graph = new Graph(getPrefixedId('save-video-graph'));
    const outputNode = graph.addNode({
        // @ts-expect-error: These nodes are not available in the OSS application
        type: 'save_runway_video',
        id: getPrefixedId('save_runway_video'),
        runway_task_id: taskId,
    });
    return { graph, outputNodeId: outputNode.id };

};
