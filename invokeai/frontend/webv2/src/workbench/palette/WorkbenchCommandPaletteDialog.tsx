import type { GalleryImage } from '@features/gallery/contracts';
import type { HotkeyDefinition } from '@workbench/hotkeys/types';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
import type { openWidgetPlacement as OpenWidgetPlacement } from '@workbench/widgetPlacementCommands';
import type { getWidgetsForRegion as GetWidgetsForRegion } from '@workbench/widgetRegistry';
import type { TFunction } from 'i18next';

import { imageIndexAvailabilityOptions } from '@features/gallery/queries';
import { flushGenerateDrafts, notifyGenerateModelSelectionCleared } from '@features/generation/react';
import { getModelsSnapshot } from '@features/models';
import { getQueueQueryScope, getQueueReadModelOptions } from '@features/queue/queries';
import { queryClient } from '@platform/query/client';
import { useQuery } from '@tanstack/react-query';
import { recallProjectPromptHistoryItem, selectProjectGenerateModel } from '@workbench/generationSettingsOrchestration';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import { getLayoutPresetCommandTitleOverrides } from '@workbench/layoutPresetSnapshots';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { useAvailableSettings } from '@workbench/settings/useAvailableSettings';
import { useNotify } from '@workbench/useNotify';
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
import { buildCatalogCommandEntries, buildOpenSettingsEntry, buildSettingsEntries, getEntryKeys } from './entries';
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
        presentWidgetTypeIds: new Set(presentWidgetTypeIds),
        t,
        titleOverrides: commandTitleOverrides,
      }),
      ...paletteContributions.map((contribution) =>
        buildExtensionPaletteEntry(contribution, extensions.commands.executeForSource)
      ),
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
      paletteContributions,
      preferences,
      presentWidgetTypeIds,
      settingsEntryDeps,
      settingsSections,
      t,
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
      createWorkflowsProvider({ openWorkflowWidget: () => openWidget('workflow'), t }),
      createBoardsProvider({
        openGalleryWidget: () => openWidget('gallery'),
        selectBoard: (boardId) => gallery.selectBoard(boardId),
        t,
      }),
      createModelsProvider({
        applyModel: (model, catalogModels) => {
          const { currentValues, projectId } = readCurrentGenerateContext();
          const result = selectProjectGenerateModel({
            currentValues,
            generation,
            model,
            models: catalogModels,
            projectId,
          });

          notifyGenerateModelSelectionCleared({
            clearedLabels: result.clearedLabels,
            locale: i18n.resolvedLanguage,
            modelName: model.name,
            notifications: notify,
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
    openWidgetPlacement,
    projectId,
    promptHistory,
    queueScope,
    readCurrentGenerateContext,
    requestQueueItemReveal,
    t,
    workbenchCommands,
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
