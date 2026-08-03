import type { ImageRecallKind } from '@workbench/image-actions/imageRecall';
import type { ResultDestination } from '@workbench/invocationContracts';

import { invalidateGallery } from '@features/gallery/queries';
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
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { submitActiveInvocation } from '@workbench/activeInvocationSubmission';
import { createLayoutPresetActivator, loadLayoutPresetWidgets } from '@workbench/layoutPresetActivation';
import { builtInLayoutPresetDescriptors, getLayoutPreset } from '@workbench/layoutPresets';
import { toggleCommandPalette } from '@workbench/palette/paletteStore';
import { openProjectSwitcher } from '@workbench/shell/topbar/projectSwitcherStore';
import { openWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchExtensions, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useEffect, useEffectEvent, useMemo } from 'react';

/** ⌥1 / ⌥2 / ⌥3 — the three shipped layout presets, in strip order. */
const layoutPresetCommands = builtInLayoutPresetDescriptors.map(({ hotkeyId, preset }) => ({
  id: `app.${hotkeyId}`,
  presetId: preset.id,
  title: `${preset.label} layout`,
}));

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
  'app.invokeToOtherDestination',
  'app.openCommandPalette',
  'app.openProjectSwitcher',
  'app.saveLayoutPreset',
  ...layoutPresetCommands.map(({ id }) => id),
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
  const queryClient = useQueryClient();
  const { layout, notifications, queue, widgets } = commands;
  const activateLayoutPreset = useMemo(
    () => createLayoutPresetActivator({ apply: layout.applyPreset, load: loadLayoutPresetWidgets }),
    [layout.applyPreset]
  );
  useInvocationTemplatesSelector((snapshot) => snapshot.status);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  /**
   * Submits the active route. `destinationOverride` applies to this run only and
   * is never written back to the project — "just this once, send it to the
   * gallery" must not silently retarget every subsequent invoke.
   */
  const submitInvocation = useEffectEvent(async (destinationOverride?: ResultDestination) => {
    await submitActiveInvocation({ commands, destinationOverride, getModels: getAvailableModels, queries });
  });

  const recallSelectedImage = useEffectEvent(async (kind: ImageRecallKind) => {
    const owner = captureAccountScope();

    try {
      const [{ executeImageRecall }, { getSelectedGalleryImage }] = await Promise.all([
        import('@workbench/image-actions/executeImageRecall'),
        import('@workbench/image-actions/selectedImage'),
      ]);

      assertAccountScopeCurrent(owner);
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
        owner,
        projectId: activeProject.id,
      });

      assertAccountScopeCurrent(owner);
      if (didRecall && queries.isActiveProject(activeProject.id)) {
        openWidgetPlacement({
          getWidgetsForRegion,
          options: { preferredRegions: ['left'] },
          typeId: 'generate',
          widgets,
        });
      }
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      throw error;
    }
  });

  useMountEffect(() =>
    commandApi.register({
      handler: toggleCommandPalette,
      id: 'app.openCommandPalette',
      title: 'Open Command Palette',
    })
  );

  useEffect(() => {
    const disposers = [
      commandApi.register({ handler: () => submitInvocation(), id: 'app.invoke', title: 'Invoke' }),
      commandApi.register({ handler: () => submitInvocation(), id: 'app.invokeFront', title: 'Invoke front' }),
      commandApi.register({
        handler: () => {
          const { destination } = queries.getSnapshot().activeProject.invocation;

          void submitInvocation(destination === 'canvas' ? 'gallery' : 'canvas');
        },
        id: 'app.invokeToOtherDestination',
        title: 'Invoke to the other destination',
      }),
      commandApi.register({
        handler: openProjectSwitcher,
        id: 'app.openProjectSwitcher',
        title: 'Open project switcher',
      }),
      commandApi.register({
        handler: () => layout.savePreset(queries.getSnapshot().activeProject.layout.presetId),
        id: 'app.saveLayoutPreset',
        title: 'Save changes to the active layout preset',
      }),
      ...layoutPresetCommands.map(({ id, presetId, title }) =>
        commandApi.register({ handler: () => void activateLayoutPreset(getLayoutPreset(presetId)), id, title })
      ),
      commandApi.register({
        handler: () => {
          const owner = captureAccountScope();

          void queueCommands.cancelCurrentItem().finally(() => {
            if (isAccountScopeCurrent(owner)) {
              return invalidateGallery(queryClient, owner);
            }
          });
        },
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
  }, [activateLayoutPreset, commandApi, commands, layout, notifications, queries, queryClient, queue, widgets]);
};
