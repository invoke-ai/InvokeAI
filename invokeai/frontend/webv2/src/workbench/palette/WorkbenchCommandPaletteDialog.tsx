import type { HotkeyDefinition } from '@workbench/hotkeys/types';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
import type { openWidgetPlacement as OpenWidgetPlacement } from '@workbench/widgetPlacementCommands';
import type { getWidgetsForRegion as GetWidgetsForRegion } from '@workbench/widgetRegistry';
import type { TFunction } from 'i18next';

import { getPromptHistoryRecallPatch } from '@features/generation/settings';
import { useModelsSnapshot } from '@features/models';
import { getQueueQueryScope, getQueueReadModelOptions } from '@features/queue/queries';
import { queryClient } from '@platform/query/client';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchExtensions } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import type { PaletteEntry, PaletteSearchProvider, SettingsEntryDeps } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';
import { buildCatalogCommandEntries, buildSettingsEntries } from './entries';
import { buildExtensionPaletteEntry, createExtensionSearchProvider } from './extensionPaletteAdapters';
import { getObjectIdentity } from './objectIdentity';
import {
  createBoardsProvider,
  createImagesProvider,
  createModelsProvider,
  createPromptHistoryProvider,
  createQueueItemsProvider,
  createWorkflowsProvider,
} from './paletteProviders';

const buildStaticAppEntries = (t: TFunction): PaletteEntry[] => [
  {
    group: 'App',
    groupLabel: t('commandPalette.groups.app'),
    id: 'app.openSettings',
    isPersistentRecent: true,
    keywords: 'preferences options',
    run: () => openWorkbenchSettings(),
    showInEmptyState: true,
    title: t('commandPalette.appEntries.openSettings'),
  },
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
  modifierKeyLabel,
  onClose,
  openWidgetPlacement,
  preferences,
  requestQueueItemReveal,
  settingsEntryDeps,
}: {
  catalog: readonly HotkeyDefinition[];
  formatHotkey: (hotkey: string) => string[];
  getWidgetsForRegion: typeof GetWidgetsForRegion;
  modifierKeyLabel: string;
  onClose: () => void;
  openWidgetPlacement: typeof OpenWidgetPlacement;
  preferences: WorkbenchPreferences;
  requestQueueItemReveal: (itemId: number) => void;
  settingsEntryDeps: SettingsEntryDeps;
}) => {
  const { i18n, t } = useTranslation();
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
        t,
      }),
      ...paletteContributions.map((contribution) =>
        buildExtensionPaletteEntry(contribution, extensions.commands.executeForSource)
      ),
      ...buildStaticAppEntries(t),
      ...buildSettingsEntries(preferences, settingsEntryDeps, t),
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
      t,
    ]
  );

  const workbenchCommands = useWorkbenchCommands();
  const searchStore = extensions.stores.search;
  const extensionSearchProviders = useSyncExternalStore(searchStore.subscribe, searchStore.list, searchStore.list);
  const promptRecallContextKey = `${getObjectIdentity(generateValues, 'generation')}:${getObjectIdentity(models, 'models')}`;
  const queueScope = useMemo(
    () => getQueueQueryScope({ projectId, queueJobsScope: preferences.queueJobsScope }),
    [preferences.queueJobsScope, projectId]
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
      createWorkflowsProvider({ openWorkflowWidget: () => openWidget('workflow'), t }),
      createBoardsProvider({
        openGalleryWidget: () => openWidget('gallery'),
        selectBoard: (boardId) => gallery.selectBoard(boardId),
        t,
      }),
      createModelsProvider({
        applyModel: (model) => widgets.patchValues('generate', { model, modelKey: model.key }, projectId),
        openGenerateWidget: () => openWidget('generate'),
        openModelManager: () => void executeCommand('app.selectModelsTab'),
        t,
      }),
      createImagesProvider({
        openGalleryWidget: () => openWidget('gallery'),
        openPreviewWidget: () => openWidget('preview'),
        selectBoard: (boardId) => gallery.selectBoard(boardId),
        selectImage: (image) => gallery.selectImage(image),
        locale: i18n.resolvedLanguage,
        t,
      }),
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
        recallContextKey: promptRecallContextKey,
        recallPrompt: (item) => {
          const patch = getPromptHistoryRecallPatch({ item, models, values: generateValues });

          if (patch) {
            widgets.patchValues('generate', patch, projectId);
          }
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
    extensions,
    generateValues,
    i18n.resolvedLanguage,
    models,
    getWidgetsForRegion,
    openWidgetPlacement,
    projectId,
    promptHistory,
    promptRecallContextKey,
    queueScope,
    requestQueueItemReveal,
    t,
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
