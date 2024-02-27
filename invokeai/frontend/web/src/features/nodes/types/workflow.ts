import { z } from 'zod';

import { zFieldIdentifier } from './field';
import { zInvocationNodeData, zNotesNodeData } from './invocation';

// #region Workflow misc
const zXYPosition = z
  .object({
    x: z.number(),
    y: z.number(),
  })
  .default({ x: 0, y: 0 });
export type XYPosition = z.infer<typeof zXYPosition>;

const zDimension = z.number().gt(0).nullish();
export type Dimension = z.infer<typeof zDimension>;

const zWorkflowCategory = z.enum(['user', 'default', 'project']);
export type WorkflowCategory = z.infer<typeof zWorkflowCategory>;
// #endregion

// #region Workflow Nodes
const zWorkflowInvocationNode = z.object({
  id: z.string().trim().min(1),
  type: z.literal('invocation'),
  data: zInvocationNodeData,
  position: zXYPosition,
});
const zWorkflowNotesNode = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  data: zNotesNodeData,
  position: zXYPosition,
});
const zWorkflowNode = z.union([zWorkflowInvocationNode, zWorkflowNotesNode]);

export type WorkflowInvocationNode = z.infer<typeof zWorkflowInvocationNode>;
export type WorkflowNotesNode = z.infer<typeof zWorkflowNotesNode>;
export type WorkflowNode = z.infer<typeof zWorkflowNode>;

export const isWorkflowInvocationNode = (val: unknown): val is WorkflowInvocationNode =>
  zWorkflowInvocationNode.safeParse(val).success;
// #endregion

// #region Workflow Edges
const zWorkflowEdgeBase = z.object({
  id: z.string().trim().min(1),
  source: z.string().trim().min(1),
  target: z.string().trim().min(1),
});
const zWorkflowEdgeDefault = zWorkflowEdgeBase.extend({
  type: z.literal('default'),
  sourceHandle: z.string().trim().min(1),
  targetHandle: z.string().trim().min(1),
});
const zWorkflowEdgeCollapsed = zWorkflowEdgeBase.extend({
  type: z.literal('collapsed'),
});
const zWorkflowEdge = z.union([zWorkflowEdgeDefault, zWorkflowEdgeCollapsed]);

export type WorkflowEdgeDefault = z.infer<typeof zWorkflowEdgeDefault>;
export type WorkflowEdgeCollapsed = z.infer<typeof zWorkflowEdgeCollapsed>;
export type WorkflowEdge = z.infer<typeof zWorkflowEdge>;
// #endregion

// #region Workflow
export const zWorkflowV3 = z.object({
  id: z.string().min(1).optional(),
  name: z.string(),
  author: z.string(),
  description: z.string(),
  version: z.string(),
  contact: z.string(),
  tags: z.string(),
  notes: z.string(),
  nodes: z.array(zWorkflowNode),
  edges: z.array(zWorkflowEdge),
  exposedFields: z.array(zFieldIdentifier),
  meta: z.object({
    category: zWorkflowCategory.default('user'),
    version: z.literal('3.0.0'),
  }),
});
export type WorkflowV3 = z.infer<typeof zWorkflowV3>;
// #endregion
