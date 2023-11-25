import { Node } from 'reactflow';
import { z } from 'zod';
import { zProgressImage } from './common';
import {
  zFieldInputInstance,
  zFieldInputTemplate,
  zFieldOutputInstance,
  zFieldOutputTemplate,
} from './field';
import { zSemVer } from './semver';

// #region InvocationTemplate
export const zInvocationTemplate = z.object({
  type: z.string(),
  title: z.string(),
  description: z.string(),
  tags: z.array(z.string().min(1)),
  inputs: z.record(zFieldInputTemplate),
  outputs: z.record(zFieldOutputTemplate),
  outputType: z.string().min(1),
  withWorkflow: z.boolean(),
  version: zSemVer,
  useCache: z.boolean(),
});
export type InvocationTemplate = z.infer<typeof zInvocationTemplate>;
// #endregion

// #region NodeData
export const zInvocationNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.string().trim().min(1),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
  embedWorkflow: z.boolean(),
  isIntermediate: z.boolean(),
  useCache: z.boolean(),
  version: zSemVer,
  inputs: z.record(zFieldInputInstance),
  outputs: z.record(zFieldOutputInstance),
});

export const zNotesNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
});
export const zCurrentImageNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('current_image'),
  label: z.string(),
  isOpen: z.boolean(),
});
export const zAnyNodeData = z.union([
  zInvocationNodeData,
  zNotesNodeData,
  zCurrentImageNodeData,
]);

export type NotesNodeData = z.infer<typeof zNotesNodeData>;
export type InvocationNodeData = z.infer<typeof zInvocationNodeData>;
export type CurrentImageNodeData = z.infer<typeof zCurrentImageNodeData>;
export type AnyNodeData = z.infer<typeof zAnyNodeData>;

export const isInvocationNode = (
  node?: Node<AnyNodeData>
): node is Node<InvocationNodeData> =>
  Boolean(node && node.type === 'invocation');
export const isNotesNode = (
  node?: Node<AnyNodeData>
): node is Node<NotesNodeData> => Boolean(node && node.type === 'notes');
export const isCurrentImageNode = (
  node?: Node<AnyNodeData>
): node is Node<CurrentImageNodeData> =>
  Boolean(node && node.type === 'current_image');
export const isInvocationNodeData = (
  node?: AnyNodeData
): node is InvocationNodeData =>
  Boolean(node && !['notes', 'current_image'].includes(node.type)); // node.type may be 'notes', 'current_image', or any invocation type
// #endregion

// #region NodeExecutionState
export const zNodeStatus = z.enum([
  'PENDING',
  'IN_PROGRESS',
  'COMPLETED',
  'FAILED',
]);
export const zNodeExecutionState = z.object({
  nodeId: z.string().trim().min(1),
  status: zNodeStatus,
  progress: z.number().nullable(),
  progressImage: zProgressImage.nullable(),
  error: z.string().nullable(),
  outputs: z.array(z.any()),
});
export type NodeExecutionState = z.infer<typeof zNodeExecutionState>;
export type NodeStatus = z.infer<typeof zNodeStatus>;
// #endregion

// #region Edges
export const zInvocationEdgeExtra = z.object({
  type: z.union([z.literal('default'), z.literal('collapsed')]),
});
export type InvocationEdgeExtra = z.infer<typeof zInvocationEdgeExtra>;
// #endregion
