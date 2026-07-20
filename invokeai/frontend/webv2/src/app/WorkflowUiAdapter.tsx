import type { WorkflowGraphPreviewPort, WorkflowUiAdapter } from '@features/workflow/react';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
import type { ReactNode } from 'react';

import { flushGenerateDrafts } from '@features/generation/react';
import { getAuthSession, subscribeAuthSession } from '@features/identity';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { nodeExecutionStore } from '@features/nodes';
import { hasPendingWorkflowQueueItem } from '@features/queue';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { createProjectedExternalStore } from '@platform/state/projectedExternalStore';
import { shallowEqual } from '@platform/state/selectors';
import { resolveAndSubmitGraphPreviewInvocation } from '@workbench/graphPreviewInvocation';
import { registerHotkeyModalLayer } from '@workbench/hotkeys';
import {
  createInvocationRouteInputSelector,
  formatRoute,
  isInvocationRouteValid,
  resolveInvocationRouteInput,
} from '@workbench/invocation';
import { markWorkbenchPerf, measureWorkbenchPerf, timeWorkbenchPerf } from '@workbench/performanceMarks';
import { getWorkbenchPreferences, subscribeWorkbenchPreferences } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
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
  workflowShowMinimap: preferences.workflowShowMinimap,
  workflowSnapToGrid: preferences.workflowSnapToGrid,
  workflowValidateConnections: preferences.workflowValidateConnections,
});

const WorkflowGraphPreviewAdapterProvider = ({ children }: { children: ReactNode }) => {
  const routeInput = useActiveProjectSelector(selectInvocationRouteInput);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const availabilityModels = modelsStatus === 'loaded' ? models : undefined;
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();

  const adapter = useMemo<WorkflowGraphPreviewPort>(
    () => ({
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
        flushGenerateDrafts();
        const { prepareCanvasInvocation } = await import('@workbench/widgets/canvas/invoke/prepareCanvasInvocation');
        return resolveAndSubmitGraphPreviewInvocation({
          commands,
          models: availabilityModels,
          prepareCanvasInvocation,
          project: queries.getSnapshot().activeProject,
          sourceId,
        });
      },
    }),
    [availabilityModels, commands, queries, routeInput]
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
        select: (snapshot) => ({
          galleryValues: getProjectWidgetValues(snapshot.activeProject, 'gallery'),
          graphHistory: snapshot.activeProject.graphHistory,
          id: snapshot.activeProject.id,
          isWorkflowRunning: hasPendingWorkflowQueueItem(snapshot.activeProject.queue.items),
          projectGraph: snapshot.activeProject.projectGraph,
          workflowValues: getProjectWidgetValues(snapshot.activeProject, 'workflow'),
        }),
        isEqual: shallowEqual,
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

  const adapter = useMemo<WorkflowUiAdapter>(
    () => ({
      capabilities,
      commands: {
        bindLibraryWorkflow: commands.workflows.bindLibraryWorkflow,
        editGraph: commands.workflows.editGraph,
        redo: commands.workflows.redo,
        replace: commands.workflows.replace,
        restoreSnapshot: commands.workflows.restoreSnapshot,
        saveSnapshot: commands.workflows.saveSnapshot,
        undo: commands.workflows.undo,
      },
      getProjectGraph: () => queries.getSnapshot().activeProject.projectGraph,
      nodeExecution: { get: nodeExecutionStore.get, subscribe: nodeExecutionStore.subscribe },
      notifications: { error: notify.error, info: notify.info, success: notify.success },
      performance: {
        mark: (name, source) => markWorkbenchPerf(name, source),
        measure: (name, start, source, end) => measureWorkbenchPerf(name, start, source, end),
        time: (name, source, callback) => timeWorkbenchPerf(name, source, callback),
      },
      preferences,
      project,
      registerModalHotkeyLayer: registerHotkeyModalLayer,
      widgets: {
        open: (options) => commands.widgets.open(options),
        patchValues: (widgetId, values) => commands.widgets.patchValues(widgetId, values),
      },
    }),
    [capabilities, commands, notify.error, notify.info, notify.success, preferences, project, queries]
  );

  return (
    <WorkflowUiProvider adapter={adapter}>
      <WorkflowGraphPreviewAdapterProvider>{children}</WorkflowGraphPreviewAdapterProvider>
    </WorkflowUiProvider>
  );
};
