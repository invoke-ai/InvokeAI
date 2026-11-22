import type { GalleryVideoItem } from '@features/gallery';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { ImageRecallKind } from '@workbench/image-actions/imageRecall';
import type { VideoRecallKind } from '@workbench/image-actions/videoRecall';
import type { ResultDestination } from '@workbench/invocationContracts';

import {
  adjustFocusedPromptAttention,
  flushGenerateDrafts,
  revealPositivePrompt,
  promptHistoryNavigation,
} from '@features/generation/react';
import { ensureModelsLoaded, getModelsSnapshot, openModelManager } from '@features/models';
import { requestQueueAction } from '@features/queue/reveal';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { submitActiveInvocation } from '@workbench/activeInvocationSubmission';
import { useWorkbenchFocus, useWorkbenchFocusTarget } from '@workbench/focusRegions';
import { builtInLayoutPresetDescriptors } from '@workbench/layoutPresets';
import { toggleCommandPalette } from '@workbench/palette/paletteStore';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { getWorkbenchPreferences } from '@workbench/settings/store';
import { openProjectSwitcher } from '@workbench/shell/topbar/projectSwitcherStore';
import { openWidgetsDialog } from '@workbench/shell/widgetsDialogStore';
import { cycleRegionWidget } from '@workbench/widgetPlacementCommands';
import { getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import { describeCanvasInvocationFailure } from '@workbench/widgets/canvas/invoke/canvasInvocationFailure';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchCommands, useWorkbenchExtensions, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { resolvePanelToggle } from '@workbench/workbenchState';
import {
  goBackToPreviousWorkspace,
  goBackToStage,
  openInWorkspace,
  toggleCenterPreview,
  type OpenInWorkspace,
} from '@workbench/workspace/workspaceNavigation';
import { useTranslation } from 'react-i18next';

import { regionFocusHotkeys, widgetCycleHotkeys } from './catalog';

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

/** The same keys on a selected video; it has no CLIP skip, and its size verb is `size`. */
const videoRecallKinds: Partial<Record<ImageRecallKind, VideoRecallKind>> = {
  all: 'all',
  dimensions: 'size',
  prompts: 'prompts',
  remix: 'remix',
  seed: 'seed',
};

export const FIRST_PARTY_APP_COMMAND_IDS = [
  ...regionFocusHotkeys.map(({ commandId }) => commandId),
  ...widgetCycleHotkeys.map(({ commandId }) => commandId),
  'app.invoke',
  'app.invokeFront',
  'app.invokeToOtherDestination',
  'app.openCommandPalette',
  'app.openProjectSwitcher',
  'app.openSettings',
  'app.saveLayoutPreset',
  ...layoutPresetCommands.map(({ id }) => id),
  'app.cancelQueueItem',
  'app.clearQueue',
  'app.selectGenerateTab',
  'app.selectCanvasTab',
  'app.selectUpscalingTab',
  'app.selectWorkflowsTab',
  'app.switchWorkflowView',
  'app.selectModelsTab',
  'app.selectQueueTab',
  'app.openWidgets',
  'app.promptHistoryPrev',
  'app.promptHistoryNext',
  'app.promptWeightUp',
  'app.promptWeightDown',
  'app.focusPrompt',
  'app.toggleLeftPanel',
  'app.toggleRightPanel',
  'app.undoLayoutChange',
  'app.redoLayoutChange',
  'app.resetPanelLayout',
  'app.togglePanels',
  'app.togglePreview',
  'app.backToStage',
  'app.backToPreviousWorkspace',
] as const;

export const FIRST_PARTY_IMAGE_RECALL_COMMAND_IDS = Object.keys(imageRecallCommands);
/** Loads the selected image's embedded workflow; shares the recall commands' selected-image lookup. */
const LOAD_WORKFLOW_COMMAND_ID = 'viewer.loadWorkflow';

export const FIRST_PARTY_COMMAND_IDS = [
  ...FIRST_PARTY_APP_COMMAND_IDS,
  ...Object.keys(imageRecallCommands),
  LOAD_WORKFLOW_COMMAND_ID,
] as const;

const getAvailableModels = () => {
  const snapshot = getModelsSnapshot();
  return snapshot.status === 'loaded' ? snapshot.models : undefined;
};

export const useRegisterFirstPartyCommands = () => {
  const commands = useWorkbenchCommands();
  const { commands: commandApi } = useWorkbenchExtensions();
  const queries = useWorkbenchQueries();
  const focus = useWorkbenchFocus();
  const getFocusTarget = useWorkbenchFocusTarget();
  const { t } = useTranslation();
  const { layout, notifications, widgets } = commands;
  const queryClient = useQueryClient();
  const navigation = { commands, focus, getWidgetsForRegion, queries };
  const open: OpenInWorkspace = (intents, options) => openInWorkspace(navigation, intents, options);
  useInvocationTemplatesSelector((snapshot) => snapshot.status);

  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  /** Submits the active route; `destinationOverride` and `prepend` apply to this run only, never persisted. */
  const submitInvocation = async (destinationOverride?: ResultDestination, prepend = false) => {
    await submitActiveInvocation({
      commands,
      destinationOverride,
      formatCanvasInvocationFailure: (failure) => describeCanvasInvocationFailure(t, failure),
      formatUpscaleMetadataFailure: (name) => t('widgets.gallery.itemActions.upscale.metadataFailed', { name }),
      getModels: getAvailableModels,
      ...(prepend ? { prepend } : {}),
      queries,
    });
  };

  const notifyNoImageSelected = () =>
    notifications.add({
      actions: [{ kind: 'open-widget', widgetId: 'gallery' }],
      kind: 'info',
      message: t('widgets.generate.selectImageOrVideoFirst'),
      title: t('widgets.generate.nothingSelected'),
    });

  const loadSelectedItemWorkflow = async () => {
    const owner = captureAccountScope();
    const [{ executeLoadMediaWorkflow }, { getSelectedGalleryItem }] = await Promise.all([
      import('@workbench/image-actions/executeImageRecall'),
      import('@workbench/image-actions/selectedImage'),
    ]);

    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    const activeProject = queries.getSnapshot().activeProject;
    const item = getSelectedGalleryItem(activeProject);

    if (!item) {
      notifyNoImageSelected();
      return;
    }

    await executeLoadMediaWorkflow({
      isProjectActive: () => queries.isActiveProject(activeProject.id),
      media: { kind: item.kind, name: item.name },
      notifications,
      openWorkflowEditor: () => open({ surface: 'stage', typeId: 'workflow' }).ok,
      t,
    });
  };

  const recallSelectedImage = async (kind: ImageRecallKind) => {
    const owner = captureAccountScope();

    try {
      const [{ executeImageRecall }, { getSelectedGalleryImage, getSelectedGalleryItem }] = await Promise.all([
        import('@workbench/image-actions/executeImageRecall'),
        import('@workbench/image-actions/selectedImage'),
      ]);

      assertAccountScopeCurrent(owner);
      const activeProject = queries.getSnapshot().activeProject;
      const image = getSelectedGalleryImage(activeProject);

      if (!image) {
        const item = getSelectedGalleryItem(activeProject);
        const videoKind = videoRecallKinds[kind];

        if (item?.kind === 'video' && videoKind) {
          await recallSelectedVideo(item, videoKind, owner, activeProject.id);
        } else {
          notifyNoImageSelected();
        }
        return;
      }

      const didRecall = await executeImageRecall({
        commands,
        // Re-read on every call: recall awaits the image record, and a snapshot from the keypress would overwrite edits
        // and recalls committed meanwhile.
        getGenerateValues: () => {
          const project = queries.getProject(activeProject.id);

          return project ? getProjectWidgetValues(project, 'generate') : {};
        },
        canvasTarget: {
          commands,
          projectId: activeProject.id,
          getProject: () =>
            queries.isActiveProject(activeProject.id) ? (queries.getProject(activeProject.id) ?? null) : null,
        },
        image,
        isCanvasFrame: activeProject.invocation.sourceId === 'canvas',
        kind,
        models: getAvailableModels() ?? [],
        owner,
        projectId: activeProject.id,
        queryClient,
        t,
      });

      assertAccountScopeCurrent(owner);
      if (didRecall && queries.isActiveProject(activeProject.id)) {
        open({ surface: 'panel', typeId: 'generate' });
      }
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      throw error;
    }
  };

  const recallSelectedVideo = async (
    item: GalleryVideoItem,
    kind: VideoRecallKind,
    owner: AccountScope,
    projectId: string
  ) => {
    // Through the entry the gallery's actions already load on every editor route: naming the module itself would split
    // it out of their chunk, at two more boot requests.
    const { executeVideoRecall } = await import('@workbench/image-actions');

    assertAccountScopeCurrent(owner);
    const didRecall = await executeVideoRecall({
      commands,
      // Re-read on every call, as for images: recall awaits the video's metadata.
      getVideoValues: () => {
        const project = queries.getProject(projectId);

        return project ? getProjectWidgetValues(project, 'video') : {};
      },
      item,
      kind,
      models: getAvailableModels() ?? [],
      owner,
      projectId,
      queryClient,
      t,
    });

    assertAccountScopeCurrent(owner);
    if (didRecall && queries.isActiveProject(projectId)) {
      open({ surface: 'panel', typeId: 'video' });
    }
  };

  /** Between the workflow's graph and its form, among the views this workspace places; it never leaves it. */
  const switchWorkflowView = () => {
    const project = queries.getSnapshot().activeProject;
    const target = getFocusTarget();
    const graphRegions = ['center', 'bottom'] as const;
    const isWorkflow = (instanceId: string) => project.widgetInstances[instanceId]?.typeId === 'workflow';
    const placedIn = (region: 'center' | 'bottom' | 'left') =>
      project.widgetRegions[region].instanceIds.find(isWorkflow);
    const visibleWorkflow = (region: 'center' | 'bottom' | 'left') => {
      const { activeInstanceId, instanceIds, isCollapsed } = project.widgetRegions[region];
      const panelOpen = region === 'center' || project.layout.panels[region === 'left' ? 'isLeftOpen' : 'isBottomOpen'];

      return panelOpen && !isCollapsed && instanceIds.includes(activeInstanceId) && isWorkflow(activeInstanceId)
        ? activeInstanceId
        : null;
    };
    const fromGraph =
      target?.kind === 'floating'
        ? isWorkflow(target.instanceId)
        : (target?.region === 'center' || target?.region === 'bottom') && visibleWorkflow(target.region) !== null;
    const showPlaced = (region: 'center' | 'bottom' | 'left', instanceId: string) => {
      // The center selects rather than fronts, so a form sharing the graph's instance stays in view beside it.
      if (region === 'center') {
        widgets.select({ projectId: project.id, region, widgetId: instanceId });
      } else {
        widgets.front({ instanceId, projectId: project.id, region });
      }
      focus.focusRegion(region, undefined, instanceId);
    };

    if (fromGraph) {
      const formId = placedIn('left');

      if (formId) {
        showPlaced('left', formId);
      } else {
        // The resolver would answer with the graph itself when it sits in the status bar's panel; the form is a rail's.
        notifications.add({
          actions: [{ kind: 'add-widget', surface: 'panel', widgetId: 'workflow' }],
          category: 'workspace-offer',
          kind: 'info',
          title: t('shell.workspace.noWorkflowForm', { lng: 'en' }),
          titleKey: 'shell.workspace.noWorkflowForm',
        });
      }
      return;
    }

    const graphRegion =
      graphRegions.find((region) => visibleWorkflow(region) !== null) ??
      graphRegions.find((region) => placedIn(region) !== undefined);
    const floatingWorkflow = Object.keys(project.floatingWidgets ?? {}).find(isWorkflow);

    if (graphRegion) {
      showPlaced(graphRegion, visibleWorkflow(graphRegion) ?? placedIn(graphRegion)!);
    } else if (floatingWorkflow) {
      widgets.revealFloating(floatingWorkflow);
      focus.focusFloating(floatingWorkflow);
    } else {
      // No graph here: the resolver offers to add the editor to this workspace.
      open({ surface: 'stage', typeId: 'workflow' }, { mode: 'stay' });
    }
  };

  useMountEffect(() =>
    commandApi.register({
      handler: toggleCommandPalette,
      id: 'app.openCommandPalette',
      title: 'Open Command Palette',
    })
  );

  useMountEffect(() => {
    const disposers = [
      ...regionFocusHotkeys.map(({ commandId, direction, title }) =>
        commandApi.register({
          handler: () => focus.focusAdjacentRegion(direction),
          id: commandId,
          title,
        })
      ),
      ...widgetCycleHotkeys.map(({ commandId, direction, title }) =>
        commandApi.register({
          handler: () => {
            const target = getFocusTarget();
            if (target?.kind !== 'region') {
              return;
            }
            const project = queries.getSnapshot().activeProject;
            const { region } = target;
            const panels = project.layout.panels;
            const isOpen =
              region === 'center' ||
              (region === 'left' ? panels.isLeftOpen : region === 'right' ? panels.isRightOpen : panels.isBottomOpen);
            if (!isOpen || project.widgetRegions[region].isCollapsed) {
              return;
            }
            const instanceId = cycleRegionWidget({
              direction,
              getWidgetsForRegion,
              project: getWidgetPlacementProject(project),
              region,
              widgets,
            });
            if (instanceId) {
              focus.focusRegion(region, undefined, instanceId);
            }
          },
          id: commandId,
          title,
        })
      ),
      commandApi.register({ handler: () => submitInvocation(), id: 'app.invoke', title: 'Invoke' }),
      commandApi.register({
        handler: () => submitInvocation(undefined, true),
        id: 'app.invokeFront',
        title: 'Invoke front',
      }),
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
        handler: () => openWorkbenchSettings(),
        id: 'app.openSettings',
        title: 'Open settings',
      }),
      commandApi.register({
        handler: () => layout.savePreset(queries.getSnapshot().activeProject.layout.presetId),
        id: 'app.saveLayoutPreset',
        title: 'Save changes to the active layout preset',
      }),
      ...layoutPresetCommands.map(({ id, presetId, title }) =>
        commandApi.register({ handler: () => void layout.activatePreset(presetId), id, title })
      ),
      // The queue menus' own actions: scoped like them, and confirming by the same rules.
      commandApi.register({
        handler: () => requestQueueAction('cancel-current'),
        id: 'app.cancelQueueItem',
        title: 'Cancel current or next queue item',
      }),
      commandApi.register({
        handler: () => requestQueueAction('cancel-all'),
        id: 'app.clearQueue',
        title: 'Cancel all queue items',
      }),
      commandApi.register({
        handler: () => open({ surface: 'panel', typeId: 'generate' }),
        id: 'app.selectGenerateTab',
        title: 'Open Generate',
      }),
      commandApi.register({
        handler: () => open({ surface: 'stage', typeId: 'canvas' }),
        id: 'app.selectCanvasTab',
        title: 'Open Canvas',
      }),
      commandApi.register({
        handler: () => open({ surface: 'panel', typeId: 'upscale' }),
        id: 'app.selectUpscalingTab',
        title: 'Open Upscale',
      }),
      commandApi.register({
        handler: () => open({ surface: 'stage', typeId: 'workflow' }),
        id: 'app.selectWorkflowsTab',
        title: 'Open Workflow editor',
      }),
      commandApi.register({
        handler: switchWorkflowView,
        id: 'app.switchWorkflowView',
        title: 'Switch workflow view',
      }),
      commandApi.register({
        handler: () => openModelManager({ projectId: queries.getSnapshot().activeProject.id }),
        id: 'app.selectModelsTab',
        title: 'Open Model Manager',
      }),
      commandApi.register({
        handler: () => open({ surface: 'panel', typeId: 'queue' }),
        id: 'app.selectQueueTab',
        title: 'Open Queue',
      }),
      commandApi.register({ handler: openWidgetsDialog, id: 'app.openWidgets', title: 'Show widgets' }),
      commandApi.register({
        // The prompt takes focus itself, once Generate shows; a region focus move would land after it.
        handler: () =>
          open(
            { surface: 'panel', typeId: 'generate' },
            { afterShown: () => revealPositivePrompt('generate'), focus: false }
          ),
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
        handler: () => adjustFocusedPromptAttention('increment', getWorkbenchPreferences().preferNumericAttentionStyle),
        id: 'app.promptWeightUp',
        title: 'Increase prompt weight',
      }),
      commandApi.register({
        handler: () => adjustFocusedPromptAttention('decrement', getWorkbenchPreferences().preferNumericAttentionStyle),
        id: 'app.promptWeightDown',
        title: 'Decrease prompt weight',
      }),
      commandApi.register({
        handler: () => {
          const toggle = resolvePanelToggle(queries.getSnapshot().activeProject.widgetRegions, ['left']);

          for (const region of toggle.regions) {
            layout.setRegionCollapsed(region, toggle.shouldCollapse);
          }
        },
        id: 'app.toggleLeftPanel',
        title: 'Toggle left panel',
      }),
      commandApi.register({
        handler: () => {
          const toggle = resolvePanelToggle(queries.getSnapshot().activeProject.widgetRegions, ['right']);

          for (const region of toggle.regions) {
            layout.setRegionCollapsed(region, toggle.shouldCollapse);
          }
        },
        id: 'app.toggleRightPanel',
        title: 'Toggle right panel',
      }),
      // Steps back or forward through the active project's layout history; with none left, the key does nothing.
      commandApi.register({ handler: () => layout.undo(), id: 'app.undoLayoutChange', title: 'Undo layout change' }),
      commandApi.register({ handler: () => layout.redo(), id: 'app.redoLayoutChange', title: 'Redo layout change' }),
      commandApi.register({
        handler: layout.reset,
        id: 'app.resetPanelLayout',
        title: 'Revert layout to saved preset',
      }),
      commandApi.register({
        handler: () => {
          const toggle = resolvePanelToggle(queries.getSnapshot().activeProject.widgetRegions, [
            'left',
            'right',
            'bottom',
          ]);

          for (const region of toggle.regions) {
            layout.setRegionCollapsed(region, toggle.shouldCollapse);
          }
        },
        id: 'app.togglePanels',
        title: 'Toggle panels',
      }),
      commandApi.register({
        handler: () => toggleCenterPreview(navigation),
        id: 'app.togglePreview',
        title: 'Toggle preview in center',
      }),
      commandApi.register({
        handler: () => goBackToStage(navigation),
        id: 'app.backToStage',
        title: 'Back to main view',
      }),
      commandApi.register({
        handler: () => goBackToPreviousWorkspace(navigation),
        id: 'app.backToPreviousWorkspace',
        title: 'Back to previous workspace',
      }),
      ...Object.entries(imageRecallCommands).map(([id, kind]) =>
        commandApi.register({ handler: () => recallSelectedImage(kind), id, title: id })
      ),
      commandApi.register({
        handler: () => loadSelectedItemWorkflow(),
        id: LOAD_WORKFLOW_COMMAND_ID,
        title: 'Load workflow from image or video',
      }),
    ];

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  });
};
