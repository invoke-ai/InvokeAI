import type { FieldType, StatefulFieldValue } from 'features/nodes/types/field';
import type {
  AnyNode,
  InvocationNodeEdge,
  InvocationTemplate,
  NodeExecutionState,
} from 'features/nodes/types/invocation';
import type { WorkflowV2 } from 'features/nodes/types/workflow';
import type { OnConnectStartParams, SelectionMode, Viewport, XYPosition } from 'reactflow';

export type NodesState = {
  _version: 1;
  nodes: AnyNode[];
  edges: InvocationNodeEdge[];
  connectionStartParams: OnConnectStartParams | null;
  connectionStartFieldType: FieldType | null;
  connectionMade: boolean;
  modifyingEdge: boolean;
  shouldShowMinimapPanel: boolean;
  shouldValidateGraph: boolean;
  shouldAnimateEdges: boolean;
  nodeOpacity: number;
  shouldSnapToGrid: boolean;
  shouldColorEdges: boolean;
  selectedNodes: string[];
  selectedEdges: string[];
  nodeExecutionStates: Record<string, NodeExecutionState>;
  viewport: Viewport;
  nodesToCopy: AnyNode[];
  edgesToCopy: InvocationNodeEdge[];
  isAddNodePopoverOpen: boolean;
  addNewNodePosition: XYPosition | null;
  selectionMode: SelectionMode;
};

export type WorkflowMode = 'edit' | 'view';
export type OriginalFieldValue = {
  fieldName: string;
  nodeId: string;
  value: StatefulFieldValue;
};

export type WorkflowsState = Omit<WorkflowV2, 'nodes' | 'edges'> & {
  _version: 1;
  isTouched: boolean;
  mode: WorkflowMode;
  originalExposedFieldValues: OriginalFieldValue[];
};

export type NodeTemplatesState = {
  templates: Record<string, InvocationTemplate>;
};
