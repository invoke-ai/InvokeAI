import type {
  FieldIdentifier,
  FieldInputTemplate,
  FieldOutputTemplate,
  StatefulFieldValue,
} from 'features/nodes/types/field';
import type {
  AnyNode,
  InvocationNodeEdge,
  InvocationTemplate,
  NodeExecutionState,
} from 'features/nodes/types/invocation';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import type { HandleType } from 'reactflow';
import type { SQLiteDirection, WorkflowRecordOrderBy } from 'services/api/types';

export type Templates = Record<string, InvocationTemplate>;
export type NodeExecutionStates = Record<string, NodeExecutionState | undefined>;

export type PendingConnection = {
  nodeId: string;
  handleId: string;
  handleType: HandleType;
  fieldTemplate: FieldInputTemplate | FieldOutputTemplate;
};

export type NodesState = {
  _version: 1;
  nodes: AnyNode[];
  edges: InvocationNodeEdge[];
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
  searchTerm: string;
  orderBy?: WorkflowRecordOrderBy;
  orderDirection: SQLiteDirection;
  categorySections: Record<string, boolean>;
};
