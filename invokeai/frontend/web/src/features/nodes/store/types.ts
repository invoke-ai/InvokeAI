import { OpenAPIV3 } from 'openapi-types';
import { Edge, Node, OnConnectStartParams, Viewport } from 'reactflow';
import {
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
  schema: OpenAPIV3.Document | null;
  nodeTemplates: Record<string, InvocationTemplate>;
  connectionStartParams: OnConnectStartParams | null;
  currentConnectionFieldType: FieldType | null;
  shouldShowFieldTypeLegend: boolean;
  shouldShowMinimapPanel: boolean;
  shouldValidateGraph: boolean;
  shouldAnimateEdges: boolean;
  nodeOpacity: number;
  shouldSnapToGrid: boolean;
  shouldColorEdges: boolean;
  selectedNodes: string[];
  selectedEdges: string[];
  workflow: Omit<Workflow, 'nodes' | 'edges'>;
  nodeExecutionStates: Record<string, NodeExecutionState>;
  viewport: Viewport;
  isReady: boolean;
};
