import type { BackendGraphContract, BackendInvocationContract } from '@features/generation/core/contracts';
import type { GenerateSettings } from '@features/generation/core/types';

import { addEdge, addNode } from './graphBuilder';

type Krea2ConditioningSettings = Pick<
  GenerateSettings,
  | 'krea2RebalanceEnabled'
  | 'krea2RebalanceMultiplier'
  | 'krea2RebalanceWeights'
  | 'krea2SeedVarianceEnabled'
  | 'krea2SeedVarianceRandomizePercent'
  | 'krea2SeedVarianceStrength'
>;

/**
 * Applies Krea-2's optional conditioning transforms in their required order.
 * `idPrefix` makes the same chain safe for any number of regional prompts.
 */
export const addKrea2ConditioningEnhancers = ({
  conditioning,
  graph,
  idPrefix,
  seed,
  settings,
}: {
  conditioning: BackendInvocationContract;
  graph: BackendGraphContract;
  idPrefix: string;
  seed: BackendInvocationContract;
  settings: Krea2ConditioningSettings;
}): BackendInvocationContract => {
  let source = conditioning;

  if (settings.krea2RebalanceEnabled) {
    const rebalance = addNode(graph, {
      id: `${idPrefix}_rebalance`,
      multiplier: settings.krea2RebalanceMultiplier,
      per_layer_weights: settings.krea2RebalanceWeights,
      type: 'krea2_conditioning_rebalance',
    });

    addEdge(graph, source, 'conditioning', rebalance, 'conditioning');
    source = rebalance;
  }

  if (settings.krea2SeedVarianceEnabled && settings.krea2SeedVarianceStrength > 0) {
    const seedVariance = addNode(graph, {
      id: `${idPrefix}_seed_variance`,
      randomize_percent: settings.krea2SeedVarianceRandomizePercent,
      strength: settings.krea2SeedVarianceStrength,
      type: 'krea2_seed_variance',
    });

    addEdge(graph, source, 'conditioning', seedVariance, 'conditioning');
    addEdge(graph, seed, 'value', seedVariance, 'variance_seed');
    source = seedVariance;
  }

  return source;
};
