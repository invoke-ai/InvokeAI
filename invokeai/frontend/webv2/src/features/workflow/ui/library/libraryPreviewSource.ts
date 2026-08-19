import type { GraphPreviewSourceState, InvocationTemplates, ProjectGraphState } from '@features/workflow/contracts';

import { compileProjectGraph } from '@features/workflow/graph';

/**
 * The graph preview dialog's data source for a library entry — the library
 * equivalent of `workbench/widget-frame/graphPreviewSource.ts`'s workflow
 * branch, minus the active project: a library entry previews its own saved
 * document, not the live project graph, so the result is never live and
 * carries no destination (the entry has not been opened into a project yet).
 *
 * `document`/`templates` come from a `'ready'` library enrichment, which only
 * exists once templates have loaded — so the try/catch below is defensive
 * (a malformed cached document), not an expected path.
 */
export const buildLibraryGraphPreviewSource = (
  document: ProjectGraphState,
  templates: InvocationTemplates
): GraphPreviewSourceState => {
  try {
    const graph = compileProjectGraph(document, templates);
    const positionHints = Object.fromEntries(document.nodes.map((node) => [node.id, node.position]));

    return {
      destinationLabel: null,
      graph,
      invalidReasons: [],
      isLive: false,
      notices: [],
      positionHints,
      // No node-count row: the side panel's summary already opens with one, and
      // adding it here rendered "Nodes" twice.
      summaryRows: [],
    };
  } catch (error) {
    return {
      destinationLabel: null,
      graph: null,
      invalidReasons: [error instanceof Error ? error.message : String(error)],
      isLive: false,
      notices: [],
      summaryRows: [],
    };
  }
};
