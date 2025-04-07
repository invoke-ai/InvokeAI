import type { HandleType } from '@xyflow/react';
import type { FieldInputTemplate, FieldOutputTemplate, StatefulFieldValue } from 'features/nodes/types/field';
import type { AnyEdge, AnyNode, InvocationTemplate, NodeExecutionState } from 'features/nodes/types/invocation';
import type { WorkflowV3 } from 'features/nodes/types/workflow';

export type Templates = Record<string, InvocationTemplate>;
export type NodeExecutionStates = Record<string, NodeExecutionState | undefined>;

export type PendingConnection = {
  nodeId: string;
  handleId: string;
  handleType: HandleType;
  fieldTemplate: FieldInputTemplate | FieldOutputTemplate;
};

export type WorkflowMode = 'edit' | 'view';

export type NodesState = {
  _version: 1;
  nodes: AnyNode[];
  edges: AnyEdge[];
  formFieldInitialValues: Record<string, StatefulFieldValue>;
} & Omit<WorkflowV3, 'nodes' | 'edges' | 'is_published'>;
