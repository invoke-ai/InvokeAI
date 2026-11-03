import type { CanvasStackForests } from '@workbench/canvas-engine/contracts';

import type { CanvasNodeInsertionAnchor } from './insertionAnchors';
import type { LayerStackKind } from './layerStacks';

import { EMPTY_STACKS } from './documentTree';
import { captureInsertionAnchor } from './insertionAnchors';

export const stackTopAnchor = (projectId: string, stack: LayerStackKind = 'raster'): CanvasNodeInsertionAnchor => ({
  afterId: null,
  beforeId: null,
  capturedEditRevision: 0,
  parentPath: [],
  projectId,
  stack,
});

/** Captures the way the engine does, against whatever `getStacks` returns at call time. */
export const createTestInsertionAnchorCapture =
  (projectId: string, getStacks: () => CanvasStackForests = () => EMPTY_STACKS) =>
  (stack: LayerStackKind, aboveId: string | null): CanvasNodeInsertionAnchor =>
    captureInsertionAnchor(getStacks(), { aboveId, editRevision: 0, projectId, stack });
