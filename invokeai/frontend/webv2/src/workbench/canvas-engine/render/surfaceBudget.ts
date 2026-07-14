import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';

import type { DerivedSurfaceCache } from './derivedSurfaceCache';
import type { LayerCacheStore } from './layerCache';

export interface SurfaceBudgetResult {
  readonly evictedDerivedLayerIds: string[];
  readonly evictedBaseLayerIds: string[];
  readonly overBudgetVisibleBaseBytes: number;
}

export const enforceSurfaceBudget = (
  base: LayerCacheStore,
  derived: DerivedSurfaceCache,
  visibleLayerIds: Iterable<string>,
  budgetBytes: number,
  diagnostics?: CanvasDiagnostics
): SurfaceBudgetResult => {
  const budget = Math.max(0, budgetBytes);
  const baseBytesBefore = base.byteSize();
  const derivedBudget = Math.max(0, budget - baseBytesBefore);
  const evictedDerivedLayerIds = derived.evictToBudget(derivedBudget);

  const baseBudget = Math.max(0, budget - derived.byteSize());
  const evictedBaseLayerIds = base.evictHidden(visibleLayerIds, baseBudget);
  const overBudgetVisibleBaseBytes = Math.max(0, base.byteSize() + derived.byteSize() - budget);
  if (overBudgetVisibleBaseBytes > 0) {
    diagnostics?.add('overBudgetVisibleBaseBytes', overBudgetVisibleBaseBytes);
  }

  return { evictedBaseLayerIds, evictedDerivedLayerIds, overBudgetVisibleBaseBytes };
};
