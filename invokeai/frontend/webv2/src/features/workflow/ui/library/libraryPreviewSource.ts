import type { GraphPreviewSourceState, InvocationTemplates, ProjectGraphState } from '@features/workflow/contracts';
import type * as GraphPreviewDialogModule from '@features/workflow/ui/graph-preview/GraphPreviewDialog';

import { Center, Portal, Spinner } from '@chakra-ui/react';
import { ForLoopGraphValidationError } from '@features/workflow/core/forLoops';
import { compileProjectGraph } from '@features/workflow/graph';
import { createDeferredModule } from '@platform/react/deferredModule';
import { useMountEffect } from '@platform/react/useMountEffect';
import { CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { LoadError } from '@platform/ui/LoadError';
import { type ComponentType, createElement, type ReactNode, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

type GraphPreviewModule = typeof GraphPreviewDialogModule;

/** The xyflow preview chunk is shared by the dialog, snapshot and in-place library graph. */
const graphPreview = createDeferredModule(() => import('@features/workflow/ui/graph-preview/GraphPreviewDialog'));
export const loadGraphPreview = graphPreview.load;
export const preloadGraphPreview = graphPreview.preload;

const PreviewLoadError = ({ onRetry }: { onRetry: () => void }) => {
  const { t } = useTranslation();
  return createElement(LoadError, {
    isRetrying: false,
    layout: 'state',
    message: t('workflowLibrary.preview.loadFailed'),
    onRetry,
  });
};

// Keep the fallback out of the preview chunk too, so it can still close when that chunk is unavailable.
const PreviewLoadDialog = ({
  isOpen,
  onOpenChange,
  onRetry,
  title,
}: {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onRetry: () => void;
  title: string;
}) => {
  const handleOpenChange = useCallback(({ open }: { open: boolean }) => onOpenChange(open), [onOpenChange]);
  return createElement(Dialog.Root, {
    open: isOpen,
    onOpenChange: handleOpenChange,
    // eslint-disable-next-line react/no-children-prop -- createElement must satisfy the required RootProps.children in this non-JSX module
    children: createElement(
      Portal,
      null,
      createElement(Dialog.Backdrop),
      createElement(
        Dialog.Positioner,
        null,
        createElement(
          Dialog.Content,
          null,
          createElement(Dialog.Header, null, createElement(Dialog.Title, null, title)),
          createElement(Dialog.Body, null, createElement(PreviewLoadError, { onRetry })),
          createElement(Dialog.CloseTrigger, { asChild: true }, createElement(CloseButton))
        )
      )
    ),
  });
};

const deferGraphPreview = <Props extends object>(
  pick: (module: GraphPreviewModule) => ComponentType<Props>,
  failure: (props: Props, retry: () => void, error: unknown) => ReactNode,
  fallback: ReactNode
) =>
  function DeferredGraphPreview(props: Props) {
    const [attempt, setAttempt] = useState(0);
    const retry = useCallback(() => setAttempt((value) => value + 1), []);
    const render = useCallback((module: GraphPreviewModule) => createElement(pick(module), props), [props]);
    const renderError = useCallback((error: unknown) => failure(props, retry, error), [props, retry]);

    return createElement(graphPreview.Deferred, { key: attempt, fallback, render, renderError });
  };

export const DeferredGraphPreviewDialog = deferGraphPreview(
  (module) => module.GraphPreviewDialog,
  (props, onRetry) =>
    createElement(PreviewLoadDialog, {
      isOpen: props.isOpen,
      onOpenChange: props.onOpenChange,
      onRetry,
      title: props.sourceLabel,
    }),
  null
);

const SnapshotLoadFailure = ({ error, onError }: { error: Error; onError: (error: Error) => void }) => {
  useMountEffect(() => {
    onError(error);
  });
  return null;
};

export const DeferredGraphPreviewSnapshot = deferGraphPreview(
  (module) => module.GraphPreviewSnapshot,
  (props, _retry, error) =>
    createElement(SnapshotLoadFailure, {
      error: error instanceof Error ? error : new Error(String(error)),
      onError: props.onError,
    }),
  null
);
export const DeferredWorkflowDocumentGraph = deferGraphPreview(
  (module) => module.WorkflowDocumentGraph,
  (_props, onRetry) => createElement(PreviewLoadError, { onRetry }),
  createElement(Center, { h: 'full', role: 'status' }, createElement(Spinner))
);

const compileLibraryGraphPreviewSource = (
  document: ProjectGraphState,
  templates: InvocationTemplates
): GraphPreviewSourceState => {
  try {
    const graph = compileProjectGraph(document, templates);
    const positionHints = Object.fromEntries(document.nodes.map((node) => [node.id, node.position]));

    return {
      destinationLabel: null,
      document,
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
      document,
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

const compiledSources = new WeakMap<
  ProjectGraphState,
  { source: GraphPreviewSourceState; templates: InvocationTemplates }
>();

/**
 * Preview the entry's saved document without active-project destination or live updates; catch malformed cached
 * data despite ready enrichment. The preview draws the document; the compiled graph is what its Graph view shows and
 * its JSON items export. Compiled once per document and templates.
 */
export const buildLibraryGraphPreviewSource = (
  document: ProjectGraphState,
  templates: InvocationTemplates
): GraphPreviewSourceState => {
  const cached = compiledSources.get(document);

  if (cached?.templates === templates) {
    return cached.source;
  }

  const source = compileLibraryGraphPreviewSource(document, templates);

  compiledSources.set(document, { source, templates });

  return source;
};
