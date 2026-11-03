import type { Rect } from '@xyflow/react';

import { getViewportForBounds } from '@xyflow/react';

import { WORKFLOW_FIT_PADDING, type WorkflowFlowInstance } from './flowInstanceStore';

// Kept out of `flowInstanceStore`, which eager library code imports: an XYFlow value import there pulls the editor's
// flow runtime and its CSS into startup.
/**
 * Fits the view to graph bounds with {@link WORKFLOW_FIT_PADDING}. XYFlow's `fitBounds` takes only one padding for
 * every side, so the viewport is resolved here from the flow's size and zoom limits.
 */
export const fitWorkflowBounds = (
  instance: WorkflowFlowInstance,
  bounds: Rect,
  flow: { height: number; maxZoom: number; minZoom: number; width: number },
  duration: number
): Promise<boolean> =>
  instance.setViewport(
    getViewportForBounds(bounds, flow.width, flow.height, flow.minZoom, flow.maxZoom, WORKFLOW_FIT_PADDING),
    { duration }
  );
