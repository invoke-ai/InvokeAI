import type { Edge, Node } from '@xyflow/react';
import { z } from 'zod';

import { zClassification, zProgressImage } from './common';
import { zFieldInputInstance, zFieldInputTemplate, zFieldOutputTemplate } from './field';
import { zSemVer } from './semver';

// #region InvocationTemplate
const _zInvocationTemplate = z.object({
  type: z.string(),
  title: z.string(),
  description: z.string(),
  tags: z.array(z.string().min(1)),
  inputs: z.record(z.string(), zFieldInputTemplate),
  outputs: z.record(z.string(), zFieldOutputTemplate),
  outputType: z.string().min(1),
  version: zSemVer,
  useCache: z.boolean(),
  nodePack: z.string().min(1).default('invokeai'),
  classification: zClassification,
});
export type InvocationTemplate = z.infer<typeof _zInvocationTemplate>;
// #endregion

// #region NodeData
export const zInvocationNodeData = z.object({
  id: z.string().trim().min(1),
  version: zSemVer,
  nodePack: z.string().min(1).default('invokeai'),
  label: z.string(),
  notes: z.string(),
  type: z.string().trim().min(1),
  inputs: z.record(z.string(), zFieldInputInstance),
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

export type NotesNodeData = z.infer<typeof zNotesNodeData>;
export type InvocationNodeData = z.infer<typeof zInvocationNodeData>;
type CurrentImageNodeData = z.infer<typeof zCurrentImageNodeData>;

const zInvocationNodeValidationSchema = z.looseObject({
  type: z.literal('invocation'),
  data: zInvocationNodeData,
});
const zInvocationNode = z.custom<Node<InvocationNodeData, 'invocation'>>(
  (val) => zInvocationNodeValidationSchema.safeParse(val).success
);
export type InvocationNode = z.infer<typeof zInvocationNode>;

const zNotesNodeValidationSchema = z.looseObject({
  type: z.literal('notes'),
  data: zNotesNodeData,
});
const zNotesNode = z.custom<Node<NotesNodeData, 'notes'>>((val) => zNotesNodeValidationSchema.safeParse(val).success);
export type NotesNode = z.infer<typeof zNotesNode>;

const zCurrentImageNodeValidationSchema = z.looseObject({
  type: z.literal('current_image'),
  data: zCurrentImageNodeData,
});
const zCurrentImageNode = z.custom<Node<CurrentImageNodeData, 'current_image'>>(
  (val) => zCurrentImageNodeValidationSchema.safeParse(val).success
);
export type CurrentImageNode = z.infer<typeof zCurrentImageNode>;

export const zAnyNode = z.union([zInvocationNode, zNotesNode, zCurrentImageNode]);
export type AnyNode = z.infer<typeof zAnyNode>;

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
const _zNodeExecutionState = z.object({
  nodeId: z.string().trim().min(1),
  status: zNodeStatus,
  progress: z.number().nullable(),
  progressImage: zProgressImage.nullable(),
  outputs: z.array(z.any()),
  error: zNodeError.nullable(),
});
export type NodeExecutionState = z.infer<typeof _zNodeExecutionState>;
// #endregion

// #region Edges
const zDefaultInvocationNodeEdgeValidationSchema = z.looseObject({
  type: z.literal('default'),
});
const zDefaultInvocationNodeEdge = z.custom<Edge<Record<string, never>, 'default'>>(
  (val) => zDefaultInvocationNodeEdgeValidationSchema.safeParse(val).success
);
export type DefaultInvocationNodeEdge = z.infer<typeof zDefaultInvocationNodeEdge>;

const zInvocationNodeEdgeCollapsedData = z.object({
  count: z.number().int().min(1),
});
const zInvocationNodeEdgeCollapsedValidationSchema = z.looseObject({
  type: z.literal('default'),
  data: zInvocationNodeEdgeCollapsedData,
});
type InvocationNodeEdgeCollapsedData = z.infer<typeof zInvocationNodeEdgeCollapsedData>;

const zCollapsedInvocationNodeEdge = z.custom<Edge<InvocationNodeEdgeCollapsedData, 'collapsed'>>(
  (val) => zInvocationNodeEdgeCollapsedValidationSchema.safeParse(val).success
);
export type CollapsedInvocationNodeEdge = z.infer<typeof zCollapsedInvocationNodeEdge>;
export const zAnyEdge = z.union([zDefaultInvocationNodeEdge, zCollapsedInvocationNodeEdge]);
export type AnyEdge = z.infer<typeof zAnyEdge>;
// #endregion

export const isBatchNodeType = (type: string) =>
  ['image_batch', 'string_batch', 'integer_batch', 'float_batch'].includes(type);

export const isGeneratorNodeType = (type: string) =>
  ['image_generator', 'string_generator', 'integer_generator', 'float_generator'].includes(type);

export const isBatchNode = (node: InvocationNode) => isBatchNodeType(node.data.type);

export const isExecutableNode = (node: InvocationNode) => {
  return !isBatchNode(node);
};
