import type {
  getQueueQueryScope as GetQueueQueryScope,
  getQueueReadModelOptions as GetQueueReadModelOptions,
} from '@features/queue/queries';
import type { HotkeyDefinition } from '@workbench/hotkeys/types';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
import type { openWidgetPlacement as OpenWidgetPlacement } from '@workbench/widgetPlacementCommands';
import type { getWidgetsForRegion as GetWidgetsForRegion } from '@workbench/widgetRegistry';

import { getPromptHistoryRecallPatch } from '@features/generation/settings';
import { useModelsSnapshot } from '@features/models';
import { requestQueueItemReveal } from '@features/queue/reveal';
import { queryClient } from '@platform/query/client';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchExtensions } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useSyncExternalStore } from 'react';

import type { PaletteEntry, PaletteSearchProvider, SettingsEntryDeps } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';
import { buildCatalogCommandEntries, buildSettingsEntries } from './entries';
import { buildExtensionPaletteEntry, createExtensionSearchProvider } from './extensionPaletteAdapters';
import {
  createBoardsProvider,
  createImagesProvider,
  createModelsProvider,
  createPromptHistoryProvider,
  createQueueItemsProvider,
  createWorkflowsProvider,
} from './paletteProviders';

const STATIC_APP_ENTRIES: PaletteEntry[] = [
  {
    group: 'App',
    id: 'app.openSettings',
    isPersistentRecent: true,
    keywords: 'preferences options',
    run: () => openWorkbenchSettings(),
    showInEmptyState: true,
    title: 'Open Settings',
  },
  {
    group: 'App',
    id: 'app.openHotkeySettings',
    isPersistentRecent: true,
    keywords: 'hotkeys keybindings shortcuts',
    run: () => openWorkbenchSettings('hotkeys'),
    title: 'Keyboard Shortcuts',
  },
];

/** Editor-only palette adapter. This module is absent from the Launchpad chunk. */
const WorkbenchCommandPaletteDialog = ({
  catalog,
  formatHotkey,
  getQueueQueryScope,
  getQueueReadModelOptions,
  getWidgetsForRegion,
  modifierKeyLabel,
  onClose,
  openWidgetPlacement,
  preferences,
  settingsEntryDeps,
}: {
  catalog: readonly HotkeyDefinition[];
  formatHotkey: (key: string) => string[];
  getQueueQueryScope: typeof GetQueueQueryScope;
  getQueueReadModelOptions: typeof GetQueueReadModelOptions;
  getWidgetsForRegion: typeof GetWidgetsForRegion;
  modifierKeyLabel: string;
  onClose: () => void;
  openWidgetPlacement: typeof OpenWidgetPlacement;
  preferences: WorkbenchPreferences;
  settingsEntryDeps: SettingsEntryDeps;
}) => {
  const extensions = useWorkbenchExtensions();
  const modelsSnapshot = useModelsSnapshot();
  const models = modelsSnapshot.status === 'loaded' ? modelsSnapshot.models : undefined;
  const projectId = useActiveProjectSelector((project) => project.id);
  const promptHistory = useActiveProjectSelector((project) => project.promptHistory);
  const generateValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'generate'));
  const presentWidgetTypeIds = useActiveProjectSelector((project) =>
    [...new Set(Object.values(project.widgetInstances).map((instance) => instance.typeId))].sort()
  );
  const paletteStore = extensions.stores.palette;
  const paletteContributions = useSyncExternalStore(paletteStore.subscribe, paletteStore.list, paletteStore.list);

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
      }),
      ...paletteContributions.map((contribution) =>
        buildExtensionPaletteEntry(contribution, extensions.commands.executeForSource)
      ),
      ...STATIC_APP_ENTRIES,
      ...buildSettingsEntries(preferences, settingsEntryDeps),
    ],
    [
      catalog,
      executeCommand,
      extensions,
      formatHotkey,
      paletteContributions,
      preferences,
      presentWidgetTypeIds,
      settingsEntryDeps,
    ]
  );

  const workbenchCommands = useWorkbenchCommands();
  const searchStore = extensions.stores.search;
  const extensionSearchProviders = useSyncExternalStore(searchStore.subscribe, searchStore.list, searchStore.list);
  const promptRecallContextKey = useMemo(() => JSON.stringify([generateValues, models]), [generateValues, models]);
  const queueScope = useMemo(
    () => getQueueQueryScope({ projectId, queueJobsScope: preferences.queueJobsScope }),
    [getQueueQueryScope, preferences.queueJobsScope, projectId]
  );
  const providers = useMemo<PaletteSearchProvider[]>(() => {
    const { gallery, widgets } = workbenchCommands;
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

    return [
      createWorkflowsProvider({ openWorkflowWidget: () => openWidget('workflow') }),
      createBoardsProvider({
        openGalleryWidget: () => openWidget('gallery'),
        selectBoard: (boardId) => gallery.selectBoard(boardId),
      }),
      createModelsProvider({
        applyModel: (model) => widgets.patchValues('generate', { model, modelKey: model.key }, projectId),
        openGenerateWidget: () => openWidget('generate'),
        openModelManager: () => void executeCommand('app.selectModelsTab'),
      }),
      createImagesProvider({
        openGalleryWidget: () => openWidget('gallery'),
        openPreviewWidget: () => openWidget('preview'),
        selectBoard: (boardId) => gallery.selectBoard(boardId),
        selectImage: (image) => gallery.selectImage(image),
      }),
      createQueueItemsProvider({
        contextKey: JSON.stringify(queueScope),
        loadQueue: () => queryClient.fetchQuery(getQueueReadModelOptions(queueScope)),
        openQueueWidget: () => openWidget('queue'),
        revealItem: requestQueueItemReveal,
      }),
      createPromptHistoryProvider({
        openGenerateWidget: () => openWidget('generate'),
        projectId,
        promptHistory,
        recallContextKey: promptRecallContextKey,
        recallPrompt: (item) => {
          const patch = getPromptHistoryRecallPatch({ item, models, values: generateValues });

          if (patch) {
            widgets.patchValues('generate', patch, projectId);
          }
        },
      }),
      ...extensionSearchProviders.map((provider) =>
        createExtensionSearchProvider(provider, extensions.commands.executeForSource)
      ),
    ];
  }, [
    executeCommand,
    extensionSearchProviders,
    extensions,
    generateValues,
    getWidgetsForRegion,
    getQueueReadModelOptions,
    models,
    openWidgetPlacement,
    projectId,
    promptHistory,
    promptRecallContextKey,
    queueScope,
    workbenchCommands,
  ]);

  return (
    <CommandPaletteDialog
      entries={entries}
      isOpen
      modifierKeyLabel={modifierKeyLabel}
      providers={providers}
      onClose={onClose}
    />
  );
};

export default WorkbenchCommandPaletteDialog;
