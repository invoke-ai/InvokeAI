import type { WorkflowModelSelectProps, WorkflowUiAdapter } from '@features/workflow/react';
import type { DiagnosticSource } from '@workbench/diagnostics/logger';
import type { InvocationSourceId } from '@workbench/invocationContracts';
import type { ComponentType, ReactNode } from 'react';

import { flushGenerateDrafts } from '@features/generation/drafts';
import { useAuthSession } from '@features/identity';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { nodeExecutionStore } from '@features/nodes';
import { hasPendingWorkflowQueueItem } from '@features/queue';
import { WorkflowUiProvider } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { resolveAndSubmitGraphPreviewInvocation } from '@workbench/graphPreviewInvocation';
import { registerHotkeyModalLayer } from '@workbench/hotkeys';
import {
  createInvocationRouteInputSelector,
  formatRoute,
  isInvocationRouteValid,
  resolveInvocationRouteInput,
} from '@workbench/invocation';
import { markWorkbenchPerf, measureWorkbenchPerf, timeWorkbenchPerf } from '@workbench/performanceMarks';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { getProjectWidgetValues } from '@workbench/widgetState';
import {
  shallowEqual,
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { lazy, useMemo } from 'react';

const ModelSelect = lazy(() => import('@features/models/react').then((module) => ({ default: module.ModelSelect })));
const WorkflowModelSelect = ModelSelect as ComponentType<WorkflowModelSelectProps>;
const selectInvocationRouteInput = createInvocationRouteInputSelector();

export const WorkflowUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const project = useActiveProjectSelector((activeProject) => ({
    activeProjectId: activeProject.id,
    galleryValues: getProjectWidgetValues(activeProject, 'gallery'),
    graphHistory: activeProject.graphHistory,
    isWorkflowRunning: hasPendingWorkflowQueueItem(activeProject.queue.items),
    projectGraph: activeProject.projectGraph,
    workflowValues: getProjectWidgetValues(activeProject, 'workflow'),
  }));
  const routeInput = useActiveProjectSelector(selectInvocationRouteInput);
  const preferences = useWorkbenchPreferenceSelector(
    (value) => ({
      reduceMotion: value.reduceMotion,
      themeId: value.themeId,
      workflowEdgeStyle: value.workflowEdgeStyle,
      workflowShowMinimap: value.workflowShowMinimap,
      workflowSnapToGrid: value.workflowSnapToGrid,
      workflowValidateConnections: value.workflowValidateConnections,
    }),
    shallowEqual
  );
  const session = useAuthSession();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const availabilityModels = modelsStatus === 'loaded' ? models : undefined;
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const notify = useNotify();

  useMountEffect(() => {
    ensureModelsLoaded();
  });

  const adapter = useMemo<WorkflowUiAdapter>(
    () => ({
      ...project,
      ModelSelect: WorkflowModelSelect,
      canUseCache: !session.multiuserEnabled || session.user?.is_admin === true,
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
      graphPreview: {
        getRoute: (sourceId) => {
          if (!sourceId) {
            return null;
          }

          const route = resolveInvocationRouteInput(
            routeInput,
            'dialog',
            { ...routeInput.invocation, sourceId: sourceId as InvocationSourceId, sourceLocked: true },
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
            sourceId: sourceId as InvocationSourceId | undefined,
          });
        },
      },
      notifications: { error: notify.error, info: notify.info, success: notify.success },
      performance: {
        mark: (name, source) => markWorkbenchPerf(name, source as DiagnosticSource),
        measure: (name, start, source, end) => measureWorkbenchPerf(name, start, source as DiagnosticSource, end),
        time: (name, source, callback) => timeWorkbenchPerf(name, source as DiagnosticSource, callback),
      },
      preferences,
      registerModalHotkeyLayer: registerHotkeyModalLayer,
      nodeExecution: { get: nodeExecutionStore.get, subscribe: nodeExecutionStore.subscribe },
      widgets: {
        open: (options) => commands.widgets.open(options as Parameters<typeof commands.widgets.open>[0]),
        patchValues: (widgetId, values) =>
          commands.widgets.patchValues(widgetId as Parameters<typeof commands.widgets.patchValues>[0], values),
        select: (options) => commands.widgets.select(options as Parameters<typeof commands.widgets.select>[0]),
      },
    }),
    [
      availabilityModels,
      commands,
      notify.error,
      notify.info,
      notify.success,
      preferences,
      project,
      queries,
      routeInput,
      session,
    ]
  );

  return <WorkflowUiProvider adapter={adapter}>{children}</WorkflowUiProvider>;
};
