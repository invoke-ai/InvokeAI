import type { ImageRecallKind } from '@workbench/image-actions/imageRecall';

import {
  adjustFocusedPromptAttention,
  flushGenerateDrafts,
  focusPositivePrompt,
  promptHistoryNavigation,
} from '@features/generation/react';
import { ensureModelsLoaded, getModelsSnapshot } from '@features/models';
import { queueCommands } from '@features/queue';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getConnectionStatus } from '@platform/transport/connectionStore';
import { isInvocationRouteValid, resolveInvocationRoute } from '@workbench/invocation';
import { submitResolvedInvocation } from '@workbench/invocationSubmit';
import { openWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchExtensions, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useEffect, useEffectEvent } from 'react';

const imageRecallCommands: Record<string, ImageRecallKind> = {
  'gallery.remix': 'remix',
  'viewer.recallAll': 'all',
  'viewer.recallPrompts': 'prompts',
  'viewer.recallSeed': 'seed',
  'viewer.remix': 'remix',
  'viewer.useSize': 'dimensions',
};

export const FIRST_PARTY_APP_COMMAND_IDS = [
  'app.invoke',
  'app.invokeFront',
  'app.cancelQueueItem',
  'app.clearQueue',
  'app.selectGenerateTab',
  'app.selectCanvasTab',
  'app.selectWorkflowsTab',
  'app.selectModelsTab',
  'app.selectQueueTab',
  'app.promptHistoryPrev',
  'app.promptHistoryNext',
  'app.promptWeightUp',
  'app.promptWeightDown',
  'app.focusPrompt',
  'app.toggleLeftPanel',
  'app.toggleRightPanel',
  'app.resetPanelLayout',
  'app.togglePanels',
] as const;

export const FIRST_PARTY_IMAGE_RECALL_COMMAND_IDS = Object.keys(imageRecallCommands);

export const FIRST_PARTY_COMMAND_IDS = [...FIRST_PARTY_APP_COMMAND_IDS, ...Object.keys(imageRecallCommands)] as const;

const getAvailableModels = () => {
  const snapshot = getModelsSnapshot();
  return snapshot.status === 'loaded' ? snapshot.models : undefined;
};

export const useRegisterFirstPartyCommands = () => {
  const commands = useWorkbenchCommands();
  const { commands: commandApi } = useWorkbenchExtensions();
  const queries = useWorkbenchQueries();
  const { layout, notifications, queue, widgets } = commands;
  useInvocationTemplatesSelector((snapshot) => snapshot.status);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const submitInvocation = useEffectEvent(async () => {
    flushGenerateDrafts();

    const { prepareCanvasInvocation } = await import('@workbench/widgets/canvas/invoke/prepareCanvasInvocation');

    const activeProject = queries.getSnapshot().activeProject;
    const resolvedRoute = resolveInvocationRoute(
      activeProject,
      'global',
      activeProject.invocation,
      getAvailableModels()
    );
    const { status } = getConnectionStatus();

    if (!isInvocationRouteValid(resolvedRoute) || status !== 'connected') {
      return;
    }

    submitResolvedInvocation({
      commands,
      models: getAvailableModels(),
      prepareCanvasInvocation,
      project: activeProject,
      route: resolvedRoute,
    });
  });

  const recallSelectedImage = useEffectEvent(async (kind: ImageRecallKind) => {
    const [{ executeImageRecall }, { getSelectedGalleryImage }] = await Promise.all([
      import('@workbench/image-actions/executeImageRecall'),
      import('@workbench/image-actions/selectedImage'),
    ]);
    const activeProject = queries.getSnapshot().activeProject;
    const image = getSelectedGalleryImage(activeProject);

    if (!image) {
      notifications.add({
        kind: 'info',
        message: 'Select an image in Gallery or Preview first.',
        title: 'No image selected',
      });
      return;
    }

    const didRecall = await executeImageRecall({
      commands,
      generateValues: getProjectWidgetValues(activeProject, 'generate'),
      image,
      kind,
      models: getAvailableModels() ?? [],
      projectId: activeProject.id,
    });

    if (didRecall && queries.isActiveProject(activeProject.id)) {
      openWidgetPlacement({
        getWidgetsForRegion,
        options: { preferredRegions: ['left'] },
        typeId: 'generate',
        widgets,
      });
    }
  });

  useEffect(() => {
    const disposers = [
      commandApi.register({ handler: submitInvocation, id: 'app.invoke', title: 'Invoke' }),
      commandApi.register({ handler: submitInvocation, id: 'app.invokeFront', title: 'Invoke front' }),
      commandApi.register({
        handler: () => void queueCommands.cancelCurrentItem().finally(queue.refreshBackendData),
        id: 'app.cancelQueueItem',
        title: 'Cancel current queue item',
      }),
      commandApi.register({
        handler: () => queue.cancelAll(queries.getSnapshot().activeProject.id),
        id: 'app.clearQueue',
        title: 'Clear queue',
      }),
      commandApi.register({
        handler: () =>
          openWidgetPlacement({
            getWidgetsForRegion,
            options: { preferredRegions: ['left'] },
            typeId: 'generate',
            widgets,
          }),
        id: 'app.selectGenerateTab',
        title: 'Select Generate tab',
      }),
      commandApi.register({
        handler: () =>
          openWidgetPlacement({
            getWidgetsForRegion,
            options: { preferredRegions: ['center'], requireCenterView: true },
            typeId: 'canvas',
            widgets,
          }),
        id: 'app.selectCanvasTab',
        title: 'Select Canvas tab',
      }),
      commandApi.register({
        handler: () =>
          openWidgetPlacement({
            getWidgetsForRegion,
            options: { preferredRegions: ['center'], requireCenterView: true },
            typeId: 'workflow',
            widgets,
          }),
        id: 'app.selectWorkflowsTab',
        title: 'Select Workflows tab',
      }),
      commandApi.register({
        handler: () =>
          (window.location.hash = `#/models?project=${encodeURIComponent(queries.getSnapshot().activeProject.id)}`),
        id: 'app.selectModelsTab',
        title: 'Select Models tab',
      }),
      commandApi.register({
        handler: () =>
          openWidgetPlacement({
            getWidgetsForRegion,
            options: { preferredRegions: ['right', 'bottom'] },
            typeId: 'queue',
            widgets,
          }),
        id: 'app.selectQueueTab',
        title: 'Select Queue tab',
      }),
      commandApi.register({
        handler: () => {
          openWidgetPlacement({
            getWidgetsForRegion,
            options: { preferredRegions: ['left'] },
            typeId: 'generate',
            widgets,
          });
          window.requestAnimationFrame(() => focusPositivePrompt());
        },
        id: 'app.focusPrompt',
        title: 'Focus prompt',
      }),
      commandApi.register({
        handler: () => {
          flushGenerateDrafts();

          const project = queries.getSnapshot().activeProject;

          promptHistoryNavigation.navigate({
            direction: -1,
            models: getAvailableModels(),
            patchValues: (values, projectId) => widgets.patchValues('generate', values, projectId),
            projectId: project.id,
            promptHistory: project.promptHistory,
            values: getProjectWidgetValues(project, 'generate'),
          });
        },
        id: 'app.promptHistoryPrev',
        title: 'Previous prompt history item',
      }),
      commandApi.register({
        handler: () => {
          flushGenerateDrafts();

          const project = queries.getSnapshot().activeProject;

          promptHistoryNavigation.navigate({
            direction: 1,
            models: getAvailableModels(),
            patchValues: (values, projectId) => widgets.patchValues('generate', values, projectId),
            projectId: project.id,
            promptHistory: project.promptHistory,
            values: getProjectWidgetValues(project, 'generate'),
          });
        },
        id: 'app.promptHistoryNext',
        title: 'Next prompt history item',
      }),
      commandApi.register({
        handler: () =>
          adjustFocusedPromptAttention(
            'increment',
            queries.getSnapshot().activeProject.settings.preferNumericAttentionStyle
          ),
        id: 'app.promptWeightUp',
        title: 'Increase prompt weight',
      }),
      commandApi.register({
        handler: () =>
          adjustFocusedPromptAttention(
            'decrement',
            queries.getSnapshot().activeProject.settings.preferNumericAttentionStyle
          ),
        id: 'app.promptWeightDown',
        title: 'Decrease prompt weight',
      }),
      commandApi.register({
        handler: () => {
          const region = queries.getSnapshot().activeProject.widgetRegions.left;

          layout.setRegionCollapsed('left', !region.isCollapsed);
        },
        id: 'app.toggleLeftPanel',
        title: 'Toggle left panel',
      }),
      commandApi.register({
        handler: () => {
          const region = queries.getSnapshot().activeProject.widgetRegions.right;

          layout.setRegionCollapsed('right', !region.isCollapsed);
        },
        id: 'app.toggleRightPanel',
        title: 'Toggle right panel',
      }),
      commandApi.register({
        handler: layout.reset,
        id: 'app.resetPanelLayout',
        title: 'Reset panel layout',
      }),
      commandApi.register({
        handler: () => {
          const { bottom, left, right } = queries.getSnapshot().activeProject.widgetRegions;
          const shouldCollapse = !left.isCollapsed || !right.isCollapsed || !bottom.isCollapsed;

          layout.setRegionCollapsed('left', shouldCollapse);
          layout.setRegionCollapsed('right', shouldCollapse);
          layout.setRegionCollapsed('bottom', shouldCollapse);
        },
        id: 'app.togglePanels',
        title: 'Toggle panels',
      }),
      ...Object.entries(imageRecallCommands).map(([id, kind]) =>
        commandApi.register({ handler: () => recallSelectedImage(kind), id, title: id })
      ),
    ];

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [commandApi, commands, layout, notifications, queries, queue, widgets]);
};
