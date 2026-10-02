import type { WorkflowGraphPreviewPort, WorkflowProjectPersistence, WorkflowUiAdapter } from '@features/workflow/react';
import type { ProjectSyncSnapshot } from '@workbench/projects/syncStore';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';
import type { ReactNode } from 'react';

import { flushGenerateDrafts } from '@features/generation/react';
import { getAuthSession, subscribeAuthSession } from '@features/identity';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { nodeExecutionStore } from '@features/nodes';
import { hasPendingWorkflowQueueItem } from '@features/queue';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { createProjectedExternalStore } from '@platform/state/projectedExternalStore';
import { shallowEqual } from '@platform/state/selectors';
import { focusOpenedWidget } from '@workbench/focusRegions';
import { resolveAndSubmitGraphPreviewInvocation } from '@workbench/graphPreviewInvocation';
import { registerHotkeyModalLayer } from '@workbench/hotkeys';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import {
  createInvocationRouteInputSelector,
  formatRoute,
  isInvocationRouteValid,
  resolveInvocationRouteInput,
} from '@workbench/invocation';
import { markWorkbenchPerf, measureWorkbenchPerf, timeWorkbenchPerf } from '@workbench/performanceMarks';
import { getProjectSyncSnapshot, subscribeProjectSync } from '@workbench/projects/syncStore';
import { getActiveProjectWorkflow } from '@workbench/projectWorkflows';
import { getWorkbenchPreferences, subscribeWorkbenchPreferences } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { getProjectWidgetValues } from '@workbench/widgetState';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchInternalStore,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { useMemo } from 'react';

const selectInvocationRouteInput = createInvocationRouteInputSelector();

const selectWorkflowPreferences = (preferences: WorkbenchPreferences) => ({
  reduceMotion: preferences.reduceMotion,
  themeId: preferences.themeId,
  workflowEdgeStyle: preferences.workflowEdgeStyle,
  workflowEdgesBehindNodes: preferences.workflowEdgesBehindNodes,
  workflowGroupNodesByCategory: preferences.workflowGroupNodesByCategory,
  workflowShowMinimap: preferences.workflowShowMinimap,
  workflowSnapToGrid: preferences.workflowSnapToGrid,
  workflowValidateConnections: preferences.workflowValidateConnections,
});

/**
 * The active project's own persistence, read from the same sync service the conflict banner uses. "Saved" means
 * the server acknowledged the document; a pending push with browser recovery is still only local.
 */
const selectProjectPersistence = ([workbench, sync]: readonly [
  WorkbenchSnapshot,
  ProjectSyncSnapshot,
]): WorkflowProjectPersistence => {
  const projectId = workbench.activeProject.id;
  const info = sync.projects[projectId];
  const autosave = workbench.autosave;
  const isUnacknowledged = info === undefined || info.revision === null || info.isPendingPush;
  const status: WorkflowProjectPersistence['status'] = info?.conflict
    ? 'conflict'
    : info?.schemaRefusal
      ? 'error'
      : autosave.status === 'error'
        ? 'error'
        : autosave.status === 'saving'
          ? 'saving'
          : autosave.status === 'pending' || isUnacknowledged
            ? 'pending'
            : 'saved';

  return {
    error: autosave.error ?? null,
    hasLocalRecovery: sync.localDraftStatus === 'ok',
    lastSavedAt: status === 'saved' ? (autosave.lastSavedAt ?? sync.lastSyncedAt) : null,
    status,
  };
};

/** One source over two stores: a change in either republishes a fresh pair, an unchanged pair keeps identity. */
const createPairedSource = <A, B>(
  a: { getSnapshot: () => A; subscribe: (listener: () => void) => () => void },
  b: { getSnapshot: () => B; subscribe: (listener: () => void) => () => void }
) => {
  const cache = { pair: [a.getSnapshot(), b.getSnapshot()] as readonly [A, B] };

  return {
    getSnapshot: (): readonly [A, B] => {
      const next = [a.getSnapshot(), b.getSnapshot()] as const;

      if (next[0] !== cache.pair[0] || next[1] !== cache.pair[1]) {
        cache.pair = next;
      }

      return cache.pair;
    },
    subscribe: (listener: () => void) => {
      const unsubscribers = [a.subscribe(listener), b.subscribe(listener)];

      return () => unsubscribers.forEach((unsubscribe) => unsubscribe());
    },
  };
};

const WorkflowGraphPreviewAdapterProvider = ({ children }: { children: ReactNode }) => {
  const routeInput = useActiveProjectSelector(selectInvocationRouteInput);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const availabilityModels = modelsStatus === 'loaded' ? models : undefined;
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const openWidget = useOpenWorkbenchWidget();

  const adapter = useMemo<WorkflowGraphPreviewPort>(
    () => ({
      focusSource: (sourceId) => {
        if (sourceId) {
          openWidget(sourceId);
        }
      },
      getRoute: (sourceId) => {
        if (!sourceId) {
          return null;
        }

        const route = resolveInvocationRouteInput(
          routeInput,
          'dialog',
          { ...routeInput.invocation, sourceId, sourceLocked: true },
          availabilityModels
        );

        return {
          canInvoke: isInvocationRouteValid(route),
          label: formatRoute(route),
          validationMessage: route.validationMessage,
        };
      },
      invoke: async (sourceId) => {
        const owner = captureAccountScope();
        flushGenerateDrafts();
        try {
          const { prepareCanvasInvocation } = await import('@workbench/widgets/canvas/invoke/prepareCanvasInvocation');

          assertAccountScopeCurrent(owner);
          return resolveAndSubmitGraphPreviewInvocation({
            commands,
            models: availabilityModels,
            owner,
            prepareCanvasInvocation,
            project: queries.getSnapshot().activeProject,
            sourceId,
          });
        } catch (error) {
          if (!isAccountScopeCurrent(owner)) {
            return false;
          }

          throw error;
        }
      },
      openDocumentInNewProject: (document, label, source) => {
        // Create first: the new project's blank placeholder is what the document takes over.
        commands.projects.create();
        commands.workflows.add(document, { label, reusePlaceholder: true, source });
        openWidget('workflow');
      },
      openWorkflowEditor: () => {
        openWidget('workflow');
      },
    }),
    [availabilityModels, commands, openWidget, queries, routeInput]
  );

  return <WorkflowGraphPreviewProvider adapter={adapter}>{children}</WorkflowGraphPreviewProvider>;
};

/** Production binding of Workflow's stable services and narrow read ports. */
export const WorkflowUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const store = useWorkbenchInternalStore();
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const notify = useNotify();

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const project = useMemo(
    () =>
      createProjectedExternalStore({
        source: store,
        select: (snapshot) => {
          const activeWorkflow = getActiveProjectWorkflow(snapshot.activeProject);

          return {
            activeWorkflow,
            activeWorkflowId: activeWorkflow.document.id,
            galleryValues: getProjectWidgetValues(snapshot.activeProject, 'gallery'),
            id: snapshot.activeProject.id,
            isWorkflowRunning: hasPendingWorkflowQueueItem(snapshot.activeProject.queue.items),
            projectGraph: activeWorkflow.document,
            workflowValues: getProjectWidgetValues(snapshot.activeProject, 'workflow'),
            workflows: snapshot.activeProject.workflows.entries,
          };
        },
        isEqual: shallowEqual,
      }),
    [store]
  );
  const persistence = useMemo(
    () =>
      createProjectedExternalStore({
        isEqual: shallowEqual,
        select: selectProjectPersistence,
        source: createPairedSource(store, { getSnapshot: getProjectSyncSnapshot, subscribe: subscribeProjectSync }),
      }),
    [store]
  );
  const preferences = useMemo(
    () =>
      createProjectedExternalStore({
        source: { getSnapshot: getWorkbenchPreferences, subscribe: subscribeWorkbenchPreferences },
        select: selectWorkflowPreferences,
        isEqual: shallowEqual,
      }),
    []
  );
  const capabilities = useMemo(
    () =>
      createProjectedExternalStore({
        source: { getSnapshot: getAuthSession, subscribe: subscribeAuthSession },
        select: (session) => ({ canUseCache: !session.multiuserEnabled || session.user?.is_admin === true }),
        isEqual: shallowEqual,
      }),
    []
  );

  const findInGallery = useFindGalleryItem();
  const adapter = useMemo<WorkflowUiAdapter>(
    () => ({
      capabilities,
      commands: {
        addWorkflow: (document, options) => commands.workflows.add(document, options),
        createWorkflow: () => commands.workflows.create(),
        duplicateWorkflow: (workflowId, copyName) => commands.workflows.duplicate(workflowId, copyName),
        editGraph: commands.workflows.editGraph,
        redo: commands.workflows.redo,
        removeWorkflow: (workflowId) => commands.workflows.remove(workflowId),
        renameWorkflow: (workflowId, name) => {
          commands.workflows.rename(workflowId, name);
        },
        replaceWorkflow: (target, document, options) => commands.workflows.replaceDocument(target, document, options),
        selectWorkflow: (workflowId) => commands.workflows.select(workflowId),
        setWorkflowSource: (target, source) =>
          commands.workflows.setSource(target.projectId, target.workflowId, source),
        undo: commands.workflows.undo,
      },
      findInGallery,
      getProjectGraph: () => getActiveProjectWorkflow(queries.getSnapshot().activeProject).document,
      nodeExecution: {
        get: nodeExecutionStore.get,
        getOrigin: nodeExecutionStore.getOrigin,
        subscribe: nodeExecutionStore.subscribe,
        subscribeOrigin: nodeExecutionStore.subscribeOrigin,
      },
      notifications: { error: notify.error, info: notify.info, success: notify.success },
      // Lazy-load manager filter state to preserve editor bundle boundaries; seed it before hash navigation.
      openAddModels: (query) => {
        void import('@features/models/launchpad').then(({ requestAddModelsSearch }) => {
          requestAddModelsSearch(query);
          window.location.hash = `#/models?project=${encodeURIComponent(queries.getSnapshot().activeProject.id)}`;
        });
      },
      performance: {
        mark: (name, source) => markWorkbenchPerf(name, source),
        measure: (name, start, source, end) => measureWorkbenchPerf(name, start, source, end),
        time: (name, source, callback) => timeWorkbenchPerf(name, source, callback),
      },
      persistence,
      preferences,
      project,
      registerModalHotkeyLayer: registerHotkeyModalLayer,
      widgets: {
        // Workflow opens widgets from its buttons: the opened widget takes focus and the region highlight.
        open: (options) => {
          commands.widgets.open(options);
          focusOpenedWidget(options.region, options.widgetId);
        },
        patchValues: (widgetId, values) => commands.widgets.patchValues(widgetId, values),
      },
    }),
    [
      capabilities,
      commands,
      findInGallery,
      notify.error,
      notify.info,
      notify.success,
      persistence,
      preferences,
      project,
      queries,
    ]
  );

  return (
    <WorkflowUiProvider adapter={adapter}>
      <WorkflowGraphPreviewAdapterProvider>{children}</WorkflowGraphPreviewAdapterProvider>
    </WorkflowUiProvider>
  );
};
