import { OpenAPIV3 } from 'openapi-types';
import { Edge, Node, OnConnectStartParams, ReactFlowInstance } from 'reactflow';
import { InvocationTemplate, InvocationValue } from '../types/types';

export type NodesState = {
  nodes: Node<InvocationValue>[];
  edges: Edge[];
  schema: OpenAPIV3.Document | null;
  invocationTemplates: Record<string, InvocationTemplate>;
  connectionStartParams: OnConnectStartParams | null;
  shouldShowGraphOverlay: boolean;
  shouldShowFieldTypeLegend: boolean;
  shouldShowMinimapPanel: boolean;
  editorInstance: ReactFlowInstance | undefined;
  progressNodeSize: { width: number; height: number };
};
