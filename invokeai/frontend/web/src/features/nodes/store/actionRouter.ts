import type { UnknownAction } from '@reduxjs/toolkit';
import { isPlainObject } from '@reduxjs/toolkit';

// Use symbols to prevent collisions with other metadata
const CANVAS_WORKFLOW_KEY = Symbol('CANVAS_WORKFLOW_KEY');
const NODES_WORKFLOW_KEY = Symbol('NODES_WORKFLOW_KEY');

type ActionWithMeta = UnknownAction & { meta?: Record<symbol, unknown> };

/**
 * Injects metadata into an action to mark it as belonging to the canvas workflow.
 */
export const injectCanvasWorkflowKey = (action: UnknownAction): void => {
  const actionWithMeta = action as ActionWithMeta;
  Object.assign(action, { meta: { ...(actionWithMeta.meta || {}), [CANVAS_WORKFLOW_KEY]: true } });
};

/**
 * Injects metadata into an action to mark it as belonging to the nodes workflow.
 */
export const injectNodesWorkflowKey = (action: UnknownAction): void => {
  const actionWithMeta = action as ActionWithMeta;
  Object.assign(action, { meta: { ...(actionWithMeta.meta || {}), [NODES_WORKFLOW_KEY]: true } });
};

/**
 * Type guard to check if an action is marked as belonging to the canvas workflow.
 */
export const isCanvasWorkflowAction = (action: UnknownAction): boolean => {
  const actionWithMeta = action as ActionWithMeta;
  return isPlainObject(actionWithMeta.meta) && actionWithMeta.meta?.[CANVAS_WORKFLOW_KEY] === true;
};

/**
 * Type guard to check if an action is marked as belonging to the nodes workflow.
 */
export const isNodesWorkflowAction = (action: UnknownAction): boolean => {
  const actionWithMeta = action as ActionWithMeta;
  return isPlainObject(actionWithMeta.meta) && actionWithMeta.meta?.[NODES_WORKFLOW_KEY] === true;
};
