import type { HandleType } from '@xyflow/react';
import { type FieldInputTemplate, type FieldOutputTemplate, zStatefulFieldValue } from 'features/nodes/types/field';
import { type InvocationTemplate, type NodeExecutionState, zAnyEdge, zAnyNode } from 'features/nodes/types/invocation';
import { zWorkflowV3 } from 'features/nodes/types/workflow';
import z from 'zod';

export type Templates = Record<string, InvocationTemplate>;
export type NodeExecutionStates = Record<string, NodeExecutionState | undefined>;

export type PendingConnection = {
  nodeId: string;
  handleId: string;
  handleType: HandleType;
  fieldTemplate: FieldInputTemplate | FieldOutputTemplate;
};

export const zWorkflowMode = z.enum(['edit', 'view']);
export type WorkflowMode = z.infer<typeof zWorkflowMode>;
export const zNodesState = z.object({
  _version: z.literal(1),
  nodes: z.array(zAnyNode),
  edges: z.array(zAnyEdge),
  formFieldInitialValues: z.record(z.string(), zStatefulFieldValue),
  ...zWorkflowV3.omit({ nodes: true, edges: true, is_published: true }).shape,
});
export type NodesState = z.infer<typeof zNodesState>;
