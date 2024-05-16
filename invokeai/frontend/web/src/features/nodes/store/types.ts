import type {
  FieldIdentifier,
  FieldInputTemplate,
  FieldOutputTemplate,
  FieldType,
  StatefulFieldValue,
} from 'features/nodes/types/field';
import type {
  AnyNode,
  InvocationNode,
  InvocationNodeEdge,
  InvocationTemplate,
  NodeExecutionState,
} from 'features/nodes/types/invocation';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import type { OnConnectStartParams, Viewport, XYPosition } from 'reactflow';

export type Templates = Record<string, InvocationTemplate>;

export type PendingConnection = {
  node: InvocationNode;
  template: InvocationTemplate;
  fieldTemplate: FieldInputTemplate | FieldOutputTemplate;
};

export type NodesState = {
  _version: 1;
  nodes: AnyNode[];
  edges: InvocationNodeEdge[];
  connectionStartParams: OnConnectStartParams | null;
  connectionStartFieldType: FieldType | null;
  connectionMade: boolean;
  modifyingEdge: boolean;
  selectedNodes: string[];
  selectedEdges: string[];
  nodeExecutionStates: Record<string, NodeExecutionState>;
  viewport: Viewport;
  isAddNodePopoverOpen: boolean;
  addNewNodePosition: XYPosition | null;
};

export type WorkflowMode = 'edit' | 'view';
export type FieldIdentifierWithValue = FieldIdentifier & {
  value: StatefulFieldValue;
};

export type WorkflowsState = Omit<WorkflowV3, 'nodes' | 'edges'> & {
  _version: 1;
  isTouched: boolean;
  mode: WorkflowMode;
  originalExposedFieldValues: FieldIdentifierWithValue[];
};
