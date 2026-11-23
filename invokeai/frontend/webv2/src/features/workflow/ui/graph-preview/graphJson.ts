import type { WorkflowPreviewGraph } from '@features/workflow/ui/contracts';

/**
 * The graph the server is sent, as the preview's Graph view and the graph preview's menu give it: what a bug report
 * needs.
 */
export const getGraphJsonText = (graph: WorkflowPreviewGraph): string =>
  JSON.stringify(graph.backendGraph ?? graph, null, 2);
