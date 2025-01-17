import type { Edge, Node } from 'reactflow';
import { z } from 'zod';

import { zClassification, zProgressImage } from './common';
import { zFieldInputInstance, zFieldInputTemplate, zFieldOutputTemplate } from './field';
import { zSemVer } from './semver';

// #region InvocationTemplate
const zInvocationTemplate = z.object({
  type: z.string(),
  title: z.string(),
  description: z.string(),
  tags: z.array(z.string().min(1)),
  inputs: z.record(zFieldInputTemplate),
  outputs: z.record(zFieldOutputTemplate),
  outputType: z.string().min(1),
  version: zSemVer,
  useCache: z.boolean(),
  nodePack: z.string().min(1).default('invokeai'),
  classification: zClassification,
});
export type InvocationTemplate = z.infer<typeof zInvocationTemplate>;
// #endregion

// #region NodeData
export const zInvocationNodeData = z.object({
  id: z.string().trim().min(1),
  version: zSemVer,
  nodePack: z.string().min(1).default('invokeai'),
  label: z.string(),
  notes: z.string(),
  type: z.string().trim().min(1),
  inputs: z.record(zFieldInputInstance),
  isOpen: z.boolean(),
  isIntermediate: z.boolean(),
  useCache: z.boolean(),
});

export const zNotesNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
});
const zCurrentImageNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('current_image'),
  label: z.string(),
  isOpen: z.boolean(),
});
const zAnyNodeData = z.union([zInvocationNodeData, zNotesNodeData, zCurrentImageNodeData]);

export type NotesNodeData = z.infer<typeof zNotesNodeData>;
export type InvocationNodeData = z.infer<typeof zInvocationNodeData>;
type CurrentImageNodeData = z.infer<typeof zCurrentImageNodeData>;
type AnyNodeData = z.infer<typeof zAnyNodeData>;

export type InvocationNode = Node<InvocationNodeData, 'invocation'>;
export type NotesNode = Node<NotesNodeData, 'notes'>;
export type CurrentImageNode = Node<CurrentImageNodeData, 'current_image'>;
export type AnyNode = Node<AnyNodeData>;

export const isInvocationNode = (node?: AnyNode | null): node is InvocationNode =>
  Boolean(node && node.type === 'invocation');
export const isNotesNode = (node?: AnyNode | null): node is NotesNode => Boolean(node && node.type === 'notes');
// #endregion

// #region NodeExecutionState
export const zNodeStatus = z.enum(['PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED']);
const zNodeError = z.object({
  error_type: z.string(),
  error_message: z.string(),
  error_traceback: z.string(),
});
const zNodeExecutionState = z.object({
  nodeId: z.string().trim().min(1),
  status: zNodeStatus,
  progress: z.number().nullable(),
  progressImage: zProgressImage.nullable(),
  outputs: z.array(z.any()),
  error: zNodeError.nullable(),
});
export type NodeExecutionState = z.infer<typeof zNodeExecutionState>;
// #endregion

// #region Edges
const zInvocationNodeEdgeExtra = z.object({
  type: z.union([z.literal('default'), z.literal('collapsed')]),
});
type InvocationNodeEdgeExtra = z.infer<typeof zInvocationNodeEdgeExtra>;
export type InvocationNodeEdge = Edge<InvocationNodeEdgeExtra>;
// #endregion

export const isBatchNode = (node: InvocationNode) => {
  switch (node.data.type) {
    case 'image_batch':
    case 'string_batch':
    case 'integer_batch':
    case 'float_batch':
      return true;
    default:
      return false;
  }
};

const isGeneratorNode = (node: InvocationNode) => {
  switch (node.data.type) {
    case 'float_generator':
    case 'integer_generator':
      return true;
    default:
      return false;
  }
};

export const isExecutableNode = (node: InvocationNode) => {
  return !isBatchNode(node) && !isGeneratorNode(node);
};
