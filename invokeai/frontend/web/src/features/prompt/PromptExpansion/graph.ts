import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectBase, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { imageDTOToImageField } from 'features/controlLayers/store/util';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

export const buildPromptExpansionGraph = ({
  state,
  imageDTO,
}: {
  state: RootState;
  imageDTO?: ImageDTO;
}): { graph: Graph; outputNodeId: string } => {
  const base = selectBase(state);
  assert(base, 'No main model found in state');

  const architecture = ['sdxl', 'sdxl-refiner'].includes(base) ? 'tag_based' : 'sentence_based';

  if (imageDTO) {
    const graph = new Graph(getPrefixedId('claude-analyze-image-graph'));
    const outputNode = graph.addNode({
      // @ts-expect-error: These nodes are not available in the OSS application
      type: 'claude_analyze_image',
      id: getPrefixedId('claude_analyze_image'),
      model_architecture: architecture,
      image: imageDTOToImageField(imageDTO),
    });
    return { graph, outputNodeId: outputNode.id };
  } else {
    const positivePrompt = selectPositivePrompt(state);
    const graph = new Graph(getPrefixedId('claude-expand-prompt-graph'));
    const outputNode = graph.addNode({
      // @ts-expect-error: These nodes are not available in the OSS application
      type: 'claude_expand_prompt',
      id: getPrefixedId('claude_expand_prompt'),
      model_architecture: architecture,
      prompt: positivePrompt,
    });
    return { graph, outputNodeId: outputNode.id };
  }
};
