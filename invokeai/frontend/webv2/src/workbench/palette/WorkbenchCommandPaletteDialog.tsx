import type { GalleryImage } from '@features/gallery/contracts';
import type { HotkeyDefinition } from '@workbench/hotkeys/types';
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
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
import { recallProjectPromptHistoryItem, selectProjectGenerateModel } from '@workbench/generationSettingsOrchestration';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import { getLayoutPresetCommandTitleOverrides } from '@workbench/layoutPresetSnapshots';
import { useProjectLibrary } from '@workbench/projects/library';
import { useProjectActions } from '@workbench/projects/useProjectActions';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { useAvailableSettings } from '@workbench/settings/useAvailableSettings';
import { useNotify } from '@workbench/useNotify';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { getProjectWidgetValues } from '@workbench/widgetState';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchExtensions,
  useWorkbenchQueries,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { isWorkspaceCommandAvailable, RESET_WORKSPACE_VIEW_COMMAND_ID } from '@workbench/workspace/structureLock';
import { useStructureLock } from '@workbench/workspace/useStructureLock';
import { useWorkspaceOpen } from '@workbench/workspace/useWorkspaceOpen';
import { useCallback, useMemo, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import type { PaletteEntry, PaletteSearchProvider, SettingsEntryDeps } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';
import { createDocsProvider } from './docsSearchProvider';
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
    () => getLayoutPresetCommandTitleOverrides(account, (name) => t('commandPalette.workspaceCommand', { name })),
    [account, t]
  );

  const open = useWorkspaceOpen();
  const widgetEntries = useMemo(() => {
    const openable = listPaletteWidgets(getWidgetsForRegion);

    return buildWidgetEntries({
      catalog,
      customHotkeys: preferences.customHotkeys,
      formatHotkey,
      show: (typeId) => void open({ typeId }),
      t,
      widgets: openable.map((widget) => ({
        label: resolveWidgetLabel(widget.manifest, t),
        typeId: widget.manifest.id,
      })),
    });
  }, [catalog, formatHotkey, getWidgetsForRegion, open, preferences.customHotkeys, t]);
  const commandStore = extensions.stores.commands;
  const commandContributions = useSyncExternalStore(commandStore.subscribe, commandStore.list, commandStore.list);
  const { canEditStructure, isCustomizing } = useStructureLock();
  // Reset view has nothing to do while the project shows the workspace as it is set up.
  const hasWorkspaceView = useActiveProjectSelector((project) => project.views[project.workspaceId] !== undefined);
  // The palette offers what the command's hotkey would run: a locked workspace hides its structural commands.
  const isHandled = useMemo(() => {
    const handled = new Set(commandContributions.map(({ id }) => id));
    const lock = { canEditStructure, isCustomizing };

    return (commandId: string) =>
      handled.has(commandId) &&
      isWorkspaceCommandAvailable(commandId, lock) &&
      (commandId !== RESET_WORKSPACE_VIEW_COMMAND_ID || hasWorkspaceView);
  }, [canEditStructure, commandContributions, hasWorkspaceView, isCustomizing]);

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
    const { gallery, generation } = workbenchCommands;
    const openWidget = (typeId: 'workflow' | 'gallery' | 'generate' | 'preview' | 'queue') =>
      open(
        typeId === 'workflow' || typeId === 'preview'
          ? { surface: 'stage', typeId }
          : typeId === 'generate' || typeId === 'queue'
            ? { surface: 'panel', typeId }
            : { typeId }
      );

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
          const readGenerateValues = (id: string) => {
            const project = workbenchQueries.getProject(id);

            return project ? getProjectWidgetValues(project, 'generate') : null;
          };

          notifyGenerateModelSwitched({
            locale: i18n.resolvedLanguage,
            model,
            notifications: notify,
            projectId,
            result: selection.result,
            revertPatch,
            settings: { patchGenerateSettings: generation.patchSettings, readGenerateValues },
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
      createDocsProvider({ t }),
    ];
  }, [
    executeCommand,
    extensionSearchProviders,
    findGalleryItem,
    isImageIndexReady,
    extensions,
    i18n.resolvedLanguage,
    notify,
    open,
    openProject,
    openProjects,
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
