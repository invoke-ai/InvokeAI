import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { GraphType } from 'features/nodes/util/graph/generation/Graph';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

type Arg = {
  state: RootState;
  imageDTO?: ImageDTO;
};

export const buildPromptExpansionGraph = ({ state, imageDTO }: Arg): GraphType => {
  const { model } = state.params;

  assert(model, 'No main model found in state');

  const architecture = ['sdxl', 'sdxl-refiner'].includes(model.base) ? 'tag_based' : 'sentence_based';

  const g = new Graph('prompt-expansion-graph');
  g.addNode({
    // @ts-expect-error: These nodes are not available in the OSS application
    type: imageDTO ? 'claude_analyze_image' : 'claude_expand_prompt',
    id: getPrefixedId('prompt_expansion'),
    model_architecture: architecture,
    ...(imageDTO && { image: imageDTO }),
    ...(!imageDTO && { prompt: state.params.positivePrompt }),
  });

  return g.getGraph();
};
