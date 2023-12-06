import {
  OnConnectStartParams,
  SelectionMode,
  Viewport,
  XYPosition,
} from 'reactflow';
import { FieldIdentifier, FieldType } from 'features/nodes/types/field';
import {
  AnyNode,
  InvocationNodeEdge,
  InvocationTemplate,
  NodeExecutionState,
} from 'features/nodes/types/invocation';
import { WorkflowV2 } from 'features/nodes/types/workflow';

export type NodesState = {
  nodes: AnyNode[];
  edges: InvocationNodeEdge[];
  nodeTemplates: Record<string, InvocationTemplate>;
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
  isReady: boolean;
  mouseOverField: FieldIdentifier | null;
  mouseOverNode: string | null;
  nodesToCopy: AnyNode[];
  edgesToCopy: InvocationNodeEdge[];
  isAddNodePopoverOpen: boolean;
  addNewNodePosition: XYPosition | null;
  selectionMode: SelectionMode;
};

export type WorkflowsState = Omit<WorkflowV2, 'nodes' | 'edges'> & {
  isTouched: boolean;
};
