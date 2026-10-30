import type { GalleryImage } from '@features/gallery/contracts';
import type { HotkeyDefinition } from '@workbench/hotkeys/types';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
import type { openWidgetPlacement as OpenWidgetPlacement } from '@workbench/widgetPlacementCommands';
import type { getWidgetsForRegion as GetWidgetsForRegion } from '@workbench/widgetRegistry';
import type { TFunction } from 'i18next';

import { imageIndexAvailabilityOptions } from '@features/gallery/queries';
import {
  flushGenerateDrafts,
  notifyGenerateModelSwitched,
  requestGenerateModelSwitchConfirmation,
} from '@features/generation/react';
import { getModelsSnapshot } from '@features/models';
import { getQueueQueryScope, getQueueReadModelOptions } from '@features/queue/queries';
import { queryClient } from '@platform/query/client';
import { useQuery } from '@tanstack/react-query';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { recallProjectPromptHistoryItem, selectProjectGenerateModel } from '@workbench/generationSettingsOrchestration';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import { getLayoutPresetCommandTitleOverrides } from '@workbench/layoutPresetSnapshots';
import { useProjectLibrary } from '@workbench/projects/library';
import { useProjectActions } from '@workbench/projects/useProjectActions';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { useAvailableSettings } from '@workbench/settings/useAvailableSettings';
import { useNotify } from '@workbench/useNotify';
import { revealOrOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { getProjectWidgetValues } from '@workbench/widgetState';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchExtensions,
  useWorkbenchQueries,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import type { PaletteEntry, PaletteSearchProvider, SettingsEntryDeps } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';
import {
  buildCatalogCommandEntries,
  buildOpenSettingsEntry,
  buildSettingsEntries,
  buildWidgetEntries,
  getEntryKeys,
} from './entries';
import { buildExtensionPaletteEntry, createExtensionSearchProvider } from './extensionPaletteAdapters';
import {
  createBoardsProvider,
  createImagesProvider,
  createModelsProvider,
  createPromptHistoryProvider,
  createQueueItemsProvider,
  createSemanticImagesProvider,
  createWorkflowsProvider,
} from './paletteProviders';
import { listPaletteWidgets } from './paletteWidgets';
import { createProjectsProvider } from './projectsProvider';

type OpenProjectName = { id: string; name: string };

const areOpenProjectNamesEqual = (left: readonly OpenProjectName[], right: readonly OpenProjectName[]): boolean =>
  left.length === right.length &&
  left.every((project, index) => project.id === right[index]?.id && project.name === right[index]?.name);

const buildStaticAppEntries = (t: TFunction, settingsKeys: string[] | undefined): PaletteEntry[] => [
  buildOpenSettingsEntry(t, () => openWorkbenchSettings(), settingsKeys),
  {
    group: 'App',
    groupLabel: t('commandPalette.groups.app'),
    id: 'app.openHotkeySettings',
    isPersistentRecent: true,
    keywords: 'hotkeys keybindings shortcuts',
    run: () => openWorkbenchSettings('hotkeys'),
    title: t('commandPalette.appEntries.keyboardShortcuts'),
  },
];

/** Editor-only palette adapter. This module is absent from the Launchpad chunk. */
const WorkbenchCommandPaletteDialog = ({
  catalog,
  formatHotkey,
  getWidgetsForRegion,
  isOpen,
  modifierKeyLabel,
  onClose,
  onExitComplete,
  openWidgetPlacement,
  preferences,
  requestQueueItemReveal,
  settingsEntryDeps,
}: {
  catalog: readonly HotkeyDefinition[];
  formatHotkey: (hotkey: string) => string[];
  getWidgetsForRegion: typeof GetWidgetsForRegion;
  isOpen: boolean;
  modifierKeyLabel: string;
  onClose: () => void;
  onExitComplete: () => void;
  openWidgetPlacement: typeof OpenWidgetPlacement;
  preferences: WorkbenchPreferences;
  requestQueueItemReveal: (itemId: number) => void;
  settingsEntryDeps: SettingsEntryDeps;
}) => {
  const { i18n, t } = useTranslation();
  const extensions = useWorkbenchExtensions();
  const notify = useNotify();
  const workbenchQueries = useWorkbenchQueries();
  const projectId = useActiveProjectSelector((project) => project.id);
  const account = useWorkbenchSelector((snapshot) => snapshot.account);
  const promptHistory = useActiveProjectSelector((project) => project.promptHistory);
  // Names only: project edits must not rebuild the providers or re-key the projects search.
  const openProjects = useWorkbenchSelector(
    (snapshot) => snapshot.projects.map(({ id, name }) => ({ id, name })),
    areOpenProjectNamesEqual
  );
  const projectLibrary = useProjectLibrary();
  const { openProject } = useProjectActions();
  const presentWidgetTypeIds = useActiveProjectSelector((project) =>
    [...new Set(Object.values(project.widgetInstances).map((instance) => instance.typeId))].sort()
  );
  const settingsSections = useAvailableSettings();
  const paletteStore = extensions.stores.palette;
  const paletteContributions = useSyncExternalStore(paletteStore.subscribe, paletteStore.list, paletteStore.list);
  const commandTitleOverrides = useMemo(
    () => getLayoutPresetCommandTitleOverrides(account, (name) => t('commandPalette.layoutPresetCommand', { name })),
    [account, t]
  );

  const focus = useWorkbenchFocus();
  const { widgets: widgetCommands } = useWorkbenchCommands();
  const widgetEntries = useMemo(() => {
    const openable = listPaletteWidgets(getWidgetsForRegion);

    return buildWidgetEntries({
      catalog,
      customHotkeys: preferences.customHotkeys,
      formatHotkey,
      show: (typeId) =>
        revealOrOpenWorkbenchWidget(
          { focus, getWidgetsForRegion, queries: workbenchQueries, widgets: widgetCommands },
          typeId
        ),
      t,
      widgets: openable.map((widget) => ({
        label: resolveWidgetLabel(widget.manifest, t),
        typeId: widget.manifest.id,
      })),
    });
  }, [
    catalog,
    focus,
    formatHotkey,
    getWidgetsForRegion,
    preferences.customHotkeys,
    t,
    widgetCommands,
    workbenchQueries,
  ]);
  const commandStore = extensions.stores.commands;
  const commandContributions = useSyncExternalStore(commandStore.subscribe, commandStore.list, commandStore.list);
  const isHandled = useMemo(() => {
    const handled = new Set(commandContributions.map(({ id }) => id));

    return (commandId: string) => handled.has(commandId);
  }, [commandContributions]);

  const executeCommand = useCallback(
    (commandId: string) => {
      const contribution = extensions.stores.commands.findLatest(
        (candidate) => candidate.id === commandId && (!candidate.source || candidate.source.projectId === projectId)
      );

      return extensions.commands.executeForSource(commandId, contribution?.source ?? null);
    },
    [extensions, projectId]
  );

  const entries = useMemo<PaletteEntry[]>(
    () => [
      ...buildCatalogCommandEntries({
        customHotkeys: preferences.customHotkeys,
        execute: executeCommand,
        catalog,
        formatHotkey,
        isHandled,
        presentWidgetTypeIds: new Set(presentWidgetTypeIds),
        t,
        titleOverrides: commandTitleOverrides,
      }),
      ...paletteContributions.map((contribution) =>
        buildExtensionPaletteEntry(contribution, extensions.commands.executeForSource)
      ),
      ...widgetEntries,
      ...buildStaticAppEntries(
        t,
        getEntryKeys(
          catalog.find((definition) => definition.id === 'app.openSettings'),
          preferences.customHotkeys,
          formatHotkey
        )
      ),
      ...buildSettingsEntries(preferences, settingsEntryDeps, t, settingsSections),
    ],
    [
      catalog,
      commandTitleOverrides,
      executeCommand,
      extensions,
      formatHotkey,
      isHandled,
      paletteContributions,
      preferences,
      presentWidgetTypeIds,
      settingsEntryDeps,
      settingsSections,
      t,
      widgetEntries,
    ]
  );

  const workbenchCommands = useWorkbenchCommands();
  const findGalleryItem = useFindGalleryItem();
  const isImageIndexReady = useQuery(imageIndexAvailabilityOptions()).data?.state === 'ready';
  const searchStore = extensions.stores.search;
  const extensionSearchProviders = useSyncExternalStore(searchStore.subscribe, searchStore.list, searchStore.list);
  const readCurrentGenerateContext = useCallback(() => {
    flushGenerateDrafts();
    const project = workbenchQueries.getSnapshot().activeProject;

    return {
      currentValues: getProjectWidgetValues(project, 'generate'),
      isCanvasFrame: project.invocation.sourceId === 'canvas',
      projectId: project.id,
    };
  }, [workbenchQueries]);
  const queueScope = useMemo(
    () => getQueueQueryScope({ projectId, queueJobsScope: preferences.queueJobsScope }),
    [preferences.queueJobsScope, projectId]
  );
  const providers = useMemo<PaletteSearchProvider[]>(() => {
    const { gallery, generation, widgets } = workbenchCommands;
    const openWidget = (typeId: 'workflow' | 'gallery' | 'generate' | 'preview' | 'queue') =>
      openWidgetPlacement({
        getWidgetsForRegion,
        options:
          typeId === 'workflow'
            ? { preferredRegions: ['center'], requireCenterView: true }
            : typeId === 'generate'
              ? { preferredRegions: ['left'] }
              : typeId === 'queue'
                ? { preferredRegions: ['right'] }
                : { preferredRegions: ['center', 'right'] },
        typeId,
        widgets,
      });

    const imageEntryDeps = {
      openPreviewWidget: () => openWidget('preview'),
      revealImage: (image: GalleryImage) => findGalleryItem({ kind: 'image', name: image.imageName }),
      selectImage: (image: GalleryImage) => gallery.selectImageInItsBoard(image),
      locale: i18n.resolvedLanguage,
      t,
    };

    return [
      createProjectsProvider({
        library: projectLibrary,
        openProject: (id, name) => void openProject(id, name),
        openProjects,
        t,
      }),
      createWorkflowsProvider({ openWorkflowWidget: () => openWidget('workflow'), t }),
      createBoardsProvider({
        openGalleryWidget: () => openWidget('gallery'),
        selectBoard: (boardId) => gallery.selectBoard(boardId),
        t,
      }),
      createModelsProvider({
        selectModel: (model, catalogModels) => {
          const { currentValues, isCanvasFrame, projectId } = readCurrentGenerateContext();
          const selection = selectProjectGenerateModel({
            currentValues,
            generation,
            isCanvasFrame,
            model,
            models: catalogModels,
            projectId,
          });

          if (selection.status === 'needsConfirmation') {
            // The project's model card asks, as it does for its own picker, once the widget below opens.
            requestGenerateModelSwitchConfirmation({ model, projectId });
            return;
          }

          if (selection.status === 'unchanged' || !selection.revertPatch) {
            return;
          }

          const { revertPatch } = selection;

          notifyGenerateModelSwitched({
            getCurrentModelKey: () => {
              const project = workbenchQueries.getProject(projectId);
              const modelKey = project ? getProjectWidgetValues(project, 'generate').modelKey : undefined;

              return typeof modelKey === 'string' ? modelKey : undefined;
            },
            locale: i18n.resolvedLanguage,
            model,
            notifications: notify,
            projectId,
            result: selection.result,
            revert: () => generation.patchSettings(revertPatch, projectId),
            t,
          });
        },
        openGenerateWidget: () => openWidget('generate'),
        openModelManager: () => void executeCommand('app.selectModelsTab'),
        t,
      }),
      createImagesProvider(imageEntryDeps),
      // Offered only on a ready index; otherwise its scope row would lead to an unexplained empty list.
      ...(isImageIndexReady ? [createSemanticImagesProvider(imageEntryDeps)] : []),
      createQueueItemsProvider({
        contextKey: queueScope.originPrefix ?? 'all-projects',
        loadQueue: () => queryClient.fetchQuery(getQueueReadModelOptions(queueScope)),
        openQueueWidget: () => openWidget('queue'),
        revealItem: requestQueueItemReveal,
        t,
      }),
      createPromptHistoryProvider({
        openGenerateWidget: () => openWidget('generate'),
        projectId,
        promptHistory,
        recallPrompt: (item) => {
          const { currentValues, projectId } = readCurrentGenerateContext();
          const modelsSnapshot = getModelsSnapshot();

          recallProjectPromptHistoryItem({
            currentValues,
            generation,
            item,
            models: modelsSnapshot.status === 'loaded' ? modelsSnapshot.models : undefined,
            projectId,
          });
        },
        t,
      }),
      ...extensionSearchProviders.map((provider) =>
        createExtensionSearchProvider(provider, extensions.commands.executeForSource)
      ),
    ];
  }, [
    executeCommand,
    extensionSearchProviders,
    findGalleryItem,
    isImageIndexReady,
    extensions,
    i18n.resolvedLanguage,
    notify,
    getWidgetsForRegion,
    openProject,
    openProjects,
    openWidgetPlacement,
    projectId,
    projectLibrary,
    promptHistory,
    queueScope,
    readCurrentGenerateContext,
    requestQueueItemReveal,
    t,
    workbenchCommands,
    workbenchQueries,
  ]);

  return (
    <CommandPaletteDialog
      entries={entries}
      isOpen={isOpen}
      modifierKeyLabel={modifierKeyLabel}
      providers={providers}
      onClose={onClose}
      onExitComplete={onExitComplete}
    />
  );
};

export default WorkbenchCommandPaletteDialog;
