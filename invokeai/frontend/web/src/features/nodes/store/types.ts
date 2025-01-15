import type {
  FieldIdentifier,
  FieldInputInstance,
  FieldInputTemplate,
  FieldOutputTemplate,
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
export type FieldIdentifierWithInstance = FieldIdentifier & {
  field: FieldInputInstance;
};

export type WorkflowsState = Omit<WorkflowV3, 'nodes' | 'edges'> & {
  _version: 2;
  isTouched: boolean;
  mode: WorkflowMode;
  originalExposedFieldValues: FieldIdentifierWithInstance[];
  searchTerm: string;
  orderBy?: WorkflowRecordOrderBy;
  orderDirection: SQLiteDirection;
  categorySections: Record<string, boolean>;
};
