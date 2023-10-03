import {
  Edge,
  Node,
  OnConnectStartParams,
  SelectionMode,
  Viewport,
  XYPosition,
} from 'reactflow';
import {
  FieldIdentifier,
  FieldType,
  InvocationEdgeExtra,
  InvocationTemplate,
  NodeData,
  NodeExecutionState,
  Workflow,
} from '../types/types';

export type NodesState = {
  nodes: Node<NodeData>[];
  edges: Edge<InvocationEdgeExtra>[];
  past: { nodes: Node<NodeData>[]; edges: Edge<InvocationEdgeExtra>[] }[];
  future: { nodes: Node<NodeData>[]; edges: Edge<InvocationEdgeExtra>[] }[];
  nodeTemplates: Record<string, InvocationTemplate>;
  connectionStartParams: OnConnectStartParams | null;
  currentConnectionFieldType: FieldType | null;
  connectionMade: boolean;
  modifyingEdge: boolean;
  shouldShowFieldTypeLegend: boolean;
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
  nodesToCopy: Node<NodeData>[];
  edgesToCopy: Edge<InvocationEdgeExtra>[];
  isAddNodePopoverOpen: boolean;
  addNewNodePosition: XYPosition | null;
  selectionMode: SelectionMode;
  workflow: Omit<Workflow, 'nodes' | 'edges'>;
};
