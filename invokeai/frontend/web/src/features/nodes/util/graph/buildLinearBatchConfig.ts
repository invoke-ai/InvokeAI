import type { RootState } from 'app/store/store';
import { generateSeeds } from 'common/util/generateSeeds';
import { range } from 'es-toolkit/compat';
import type { SeedBehaviour } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import type { Graph, GraphUIContract, GraphUIInput } from 'features/nodes/util/graph/generation/Graph';
import { API_BASE_MODELS, VIDEO_BASE_MODELS } from 'features/parameters/types/constants';
import type { components } from 'services/api/schema';
import type { BaseModelType, Batch, EnqueueBatchArg, Invocation } from 'services/api/types';

const getExtendedPrompts = (arg: {
  seedBehaviour: SeedBehaviour;
  iterations: number;
  prompts: string[];
  base: BaseModelType;
}): string[] => {
  const { seedBehaviour, iterations, prompts, base } = arg;
  // Normally, the seed behaviour implicity determines the batch size. But when we use models without seeds (like
  // ChatGPT 4o) in conjunction with the per-prompt seed behaviour, we lose out on that implicit batch size. To rectify
  // this, we need to create a batch of the right size by repeating the prompts.
  if (seedBehaviour === 'PER_PROMPT' || API_BASE_MODELS.includes(base) || VIDEO_BASE_MODELS.includes(base)) {
    return range(iterations).flatMap(() => prompts);
  }
  return prompts;
};

export const prepareLinearUIBatch = (arg: {
  state: RootState;
  g: Graph;
  prepend: boolean;
  base: BaseModelType;
  positivePromptNode: Invocation<'string'>;
  seedNode?: Invocation<'integer'>;
  origin: string;
  destination: string;
}): EnqueueBatchArg => {
  const { state, g, base, prepend, positivePromptNode, seedNode, origin, destination } = arg;
  const { iterations, shouldRandomizeSeed, seed } = state.params;
  const { prompts, seedBehaviour } = state.dynamicPrompts;

  const graphWithContract = g.getGraph();
  const uiContract = graphWithContract.ui;

  const data: Batch['data'] = [];
  const firstBatchDatumList: components['schemas']['BatchDatum'][] = [];
  const secondBatchDatumList: components['schemas']['BatchDatum'][] = [];

  const resolvePromptInput = (contract?: GraphUIContract): { nodePath: string; fieldName: string } => {
    if (!contract) {
      return { nodePath: positivePromptNode.id, fieldName: 'value' };
    }
    const preferredId = contract.primary_input;
    const promptInput = preferredId
      ? contract.inputs.find((i) => i.id === preferredId)
      : contract.inputs.find((i) => i.kind === 'string');
    if (promptInput) {
      return { nodePath: promptInput.node_id, fieldName: promptInput.field };
    }
    return { nodePath: positivePromptNode.id, fieldName: 'value' };
  };

  const resolveSeedInput = (contract?: GraphUIContract): GraphUIInput | undefined => {
    if (!contract) {
      return undefined;
    }
    const preferredId = contract.inputs.find((i) => i.kind === 'seed');
    return preferredId;
  };

  const promptTarget = resolvePromptInput(uiContract);
  const seedTarget = resolveSeedInput(uiContract);

  const seedNodePath = seedTarget?.node_id ?? seedNode?.id;
  const seedFieldName = seedTarget?.field ?? 'value';

  // add seeds first to ensure the output order groups the prompts
  if (seedNodePath && seedBehaviour === 'PER_PROMPT') {
    const seeds = generateSeeds({
      count: prompts.length * iterations,
      start: shouldRandomizeSeed ? undefined : seed,
    });

    firstBatchDatumList.push({
      node_path: seedNodePath,
      field_name: seedFieldName,
      items: seeds,
    });
  } else if (seedNodePath && seedBehaviour === 'PER_ITERATION') {
    // seedBehaviour = SeedBehaviour.PerRun
    const seeds = generateSeeds({
      count: iterations,
      start: shouldRandomizeSeed ? undefined : seed,
    });

    secondBatchDatumList.push({
      node_path: seedNodePath,
      field_name: seedFieldName,
      items: seeds,
    });
    data.push(secondBatchDatumList);
  }

  const extendedPrompts = getExtendedPrompts({ seedBehaviour, iterations, prompts, base });

  // zipped batch of prompts
  firstBatchDatumList.push({
    node_path: promptTarget.nodePath,
    field_name: promptTarget.fieldName,
    items: extendedPrompts,
  });

  data.push(firstBatchDatumList);

  const enqueueBatchArg: EnqueueBatchArg = {
    prepend,
    batch: {
      graph: graphWithContract,
      runs: 1,
      data,
      origin,
      destination,
    },
  };

  return enqueueBatchArg;
};
