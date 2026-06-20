import { getConnectionStatus } from '@workbench/backend/connectionStore';
import { commandApi } from '@workbench/extensions/extensionApi';
import { cancelCurrentQueueItem } from '@workbench/generation/api';
import { executeImageRecall, getSelectedGalleryImage, type ImageRecallKind } from '@workbench/image-actions';
import { isInvocationRouteValid, resolveInvocationRoute } from '@workbench/invocation';
import { ensureModelsLoaded, useModelsSnapshot } from '@workbench/models/modelsStore';
import { openWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { focusPositivePrompt } from '@workbench/widgets/generate/promptFields';
import { adjustFocusedPromptAttention } from '@workbench/widgets/generate/promptFields/promptAttentionHotkeys';
import { navigatePromptHistory } from '@workbench/widgets/generate/promptFields/promptHistoryNavigation';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProject, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useInvocationTemplatesSnapshot } from '@workbench/workflows/templates';
import { useEffect, useEffectEvent, useRef } from 'react';

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

export const useRegisterFirstPartyCommands = () => {
  const project = useActiveProject();
  const dispatch = useWorkbenchDispatch();
  const { models, status: modelsStatus } = useModelsSnapshot();
  const availabilityModels = modelsStatus === 'loaded' ? models : undefined;
  const projectRef = useRef(project);
  const modelsRef = useRef(availabilityModels);

  useInvocationTemplatesSnapshot();

  projectRef.current = project;
  modelsRef.current = availabilityModels;

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  const submitInvocation = useEffectEvent(() => {
    const activeProject = projectRef.current;
    const resolvedRoute = resolveInvocationRoute(activeProject, 'global', activeProject.invocation, modelsRef.current);
    const { status } = getConnectionStatus();

    if (!isInvocationRouteValid(resolvedRoute) || status !== 'connected') {
      return;
    }

    dispatch({
      backendSupportsCancellation: true,
      models: modelsRef.current,
      route: resolvedRoute,
      type: 'submitResolvedInvocationSnapshot',
    });
  });

  const recallSelectedImage = useEffectEvent(async (kind: ImageRecallKind) => {
    const activeProject = projectRef.current;
    const image = getSelectedGalleryImage(activeProject);

    if (!image) {
      dispatch({
        kind: 'info',
        message: 'Select an image in Gallery or Preview first.',
        title: 'No image selected',
        type: 'recordNotice',
      });
      return;
    }

    const didRecall = await executeImageRecall({
      dispatch,
      generateValues: getProjectWidgetValues(activeProject, 'generate'),
      image,
      kind,
      models: modelsRef.current ?? [],
    });

    if (didRecall) {
      openWidgetPlacement({
        dispatch,
        getWidgetsForRegion,
        options: { preferredRegions: ['left'] },
        typeId: 'generate',
      });
    }
  });

  useEffect(() => {
    const disposers = [
      commandApi.register({ handler: submitInvocation, id: 'app.invoke', title: 'Invoke' }),
      commandApi.register({ handler: submitInvocation, id: 'app.invokeFront', title: 'Invoke front' }),
      commandApi.register({
        handler: () => void cancelCurrentQueueItem().finally(() => dispatch({ type: 'refreshBackendData' })),
        id: 'app.cancelQueueItem',
        title: 'Cancel current queue item',
      }),
      commandApi.register({
        handler: () => dispatch({ projectId: projectRef.current.id, type: 'cancelAllQueueItems' }),
        id: 'app.clearQueue',
        title: 'Clear queue',
      }),
      commandApi.register({
        handler: () =>
          openWidgetPlacement({
            dispatch,
            getWidgetsForRegion,
            options: { preferredRegions: ['left'] },
            typeId: 'generate',
          }),
        id: 'app.selectGenerateTab',
        title: 'Select Generate tab',
      }),
      commandApi.register({
        handler: () =>
          openWidgetPlacement({
            dispatch,
            getWidgetsForRegion,
            options: { preferredRegions: ['center'], requireCenterView: true },
            typeId: 'canvas',
          }),
        id: 'app.selectCanvasTab',
        title: 'Select Canvas tab',
      }),
      commandApi.register({
        handler: () =>
          openWidgetPlacement({
            dispatch,
            getWidgetsForRegion,
            options: { preferredRegions: ['center'], requireCenterView: true },
            typeId: 'workflow',
          }),
        id: 'app.selectWorkflowsTab',
        title: 'Select Workflows tab',
      }),
      commandApi.register({
        handler: () => (window.location.hash = '#/models'),
        id: 'app.selectModelsTab',
        title: 'Select Models tab',
      }),
      commandApi.register({
        handler: () =>
          openWidgetPlacement({
            dispatch,
            getWidgetsForRegion,
            options: { preferredRegions: ['right', 'bottom'] },
            typeId: 'queue',
          }),
        id: 'app.selectQueueTab',
        title: 'Select Queue tab',
      }),
      commandApi.register({
        handler: () => {
          openWidgetPlacement({
            dispatch,
            getWidgetsForRegion,
            options: { preferredRegions: ['left'] },
            typeId: 'generate',
          });
          window.requestAnimationFrame(() => focusPositivePrompt());
        },
        id: 'app.focusPrompt',
        title: 'Focus prompt',
      }),
      commandApi.register({
        handler: () =>
          navigatePromptHistory({
            direction: -1,
            dispatch,
            models: modelsRef.current,
            project: projectRef.current,
          }),
        id: 'app.promptHistoryPrev',
        title: 'Previous prompt history item',
      }),
      commandApi.register({
        handler: () =>
          navigatePromptHistory({
            direction: 1,
            dispatch,
            models: modelsRef.current,
            project: projectRef.current,
          }),
        id: 'app.promptHistoryNext',
        title: 'Next prompt history item',
      }),
      commandApi.register({
        handler: () =>
          adjustFocusedPromptAttention('increment', projectRef.current.settings.preferNumericAttentionStyle),
        id: 'app.promptWeightUp',
        title: 'Increase prompt weight',
      }),
      commandApi.register({
        handler: () =>
          adjustFocusedPromptAttention('decrement', projectRef.current.settings.preferNumericAttentionStyle),
        id: 'app.promptWeightDown',
        title: 'Decrease prompt weight',
      }),
      commandApi.register({
        handler: () => {
          const region = projectRef.current.widgetRegions.left;

          dispatch({ isCollapsed: !region.isCollapsed, region: 'left', type: 'setRegionWidgetCollapsed' });
        },
        id: 'app.toggleLeftPanel',
        title: 'Toggle left panel',
      }),
      commandApi.register({
        handler: () => {
          const region = projectRef.current.widgetRegions.right;

          dispatch({ isCollapsed: !region.isCollapsed, region: 'right', type: 'setRegionWidgetCollapsed' });
        },
        id: 'app.toggleRightPanel',
        title: 'Toggle right panel',
      }),
      commandApi.register({
        handler: () => dispatch({ type: 'resetActiveLayout' }),
        id: 'app.resetPanelLayout',
        title: 'Reset panel layout',
      }),
      commandApi.register({
        handler: () => {
          const { bottom, left, right } = projectRef.current.widgetRegions;
          const shouldCollapse = !left.isCollapsed || !right.isCollapsed || !bottom.isCollapsed;

          dispatch({ isCollapsed: shouldCollapse, region: 'left', type: 'setRegionWidgetCollapsed' });
          dispatch({ isCollapsed: shouldCollapse, region: 'right', type: 'setRegionWidgetCollapsed' });
          dispatch({ isCollapsed: shouldCollapse, region: 'bottom', type: 'setRegionWidgetCollapsed' });
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
  }, [dispatch]);
};
