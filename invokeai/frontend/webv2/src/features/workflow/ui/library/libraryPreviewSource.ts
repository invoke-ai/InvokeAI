import type { GraphPreviewSourceState, InvocationTemplates, ProjectGraphState } from '@features/workflow/contracts';
import type * as GraphPreviewDialogModule from '@features/workflow/ui/graph-preview/GraphPreviewDialog';

import { ForLoopGraphValidationError } from '@features/workflow/core/forLoops';
import { compileProjectGraph } from '@features/workflow/graph';
import { type ComponentType, createElement, lazy, useState } from 'react';

type GraphPreviewModule = typeof GraphPreviewDialogModule;

let loadedGraphPreview: GraphPreviewModule | null = null;
let graphPreviewModule: Promise<GraphPreviewModule> | null = null;

/** The preview chunk (xyflow and its stylesheet), kept out of the library and publication chunks until needed. */
export const loadGraphPreview = (): Promise<GraphPreviewModule> =>
  (graphPreviewModule ??= import('@features/workflow/ui/graph-preview/GraphPreviewDialog').then(
    (module) => {
      loadedGraphPreview = module;
      return module;
    },
    (error: unknown) => {
      graphPreviewModule = null;
      throw error;
    }
  ));

/** Starts the preview chunk when a dialog that can open the preview mounts. */
export const preloadGraphPreview = (): void => {
  void loadGraphPreview().catch(() => {});
};

/**
 * Renders the loaded component directly and falls back to `lazy` only before the chunk arrives. A first `lazy` render
 * always suspends; an open dialog revealed by Suspense gets its effects replayed in StrictMode after its focus trap
 * started, and the library's trap then took focus back and dismissed the preview.
 */
const deferGraphPreview = <Props extends object>(pick: (module: GraphPreviewModule) => ComponentType<Props>) => {
  const Lazy = lazy(() => loadGraphPreview().then((module) => ({ default: pick(module) })));

  return function DeferredGraphPreview(props: Props) {
    // Chosen once per mount: switching from `Lazy` to the loaded component would remount it.
    const [component] = useState(() => (loadedGraphPreview ? pick(loadedGraphPreview) : Lazy));

    return createElement(component, props);
  };
};

export const DeferredGraphPreviewDialog = deferGraphPreview((module) => module.GraphPreviewDialog);
export const DeferredGraphPreviewSnapshot = deferGraphPreview((module) => module.GraphPreviewSnapshot);

/**
 * Preview the entry's saved document without active-project destination or live updates; catch malformed cached
 * data despite ready enrichment.
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
      summaryRows: [],
    };
  } catch (error) {
    return {
      destinationLabel: null,
      graph: null,
      invalidReasons: [
        error instanceof ForLoopGraphValidationError
          ? error.reason
          : error instanceof Error
            ? error.message
            : String(error),
      ],
      isLive: false,
      notices: [],
      summaryRows: [],
    };
  }
};
