import InvocationCollapsedEdge from './edges/InvocationCollapsedEdge';
import InvocationDefaultEdge from './edges/InvocationDefaultEdge';

export const edgeTypes = {
  collapsed: InvocationCollapsedEdge,
  default: InvocationDefaultEdge,
  loop_linkage: InvocationDefaultEdge,
} as const;
