import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { patchWorkbenchPreferences, useWorkbenchPreferences } from '@workbench/settings/store';
import { openWidgetPlacement } from '@workbench/widgetPlacementCommands';
import { getWidgetsForRegion } from '@workbench/widgetRegistry';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchExtensions,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useSyncExternalStore } from 'react';

import type { PaletteEntry, PaletteSearchProvider } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';
import { buildCatalogCommandEntries, buildSettingsEntries } from './entries';
import { createBoardsProvider, createPromptHistoryProvider, createWorkflowsProvider } from './paletteProviders';
import { closeCommandPalette, commandPaletteStore } from './paletteStore';

/**
 * Editor host: aggregates the hotkey catalog, extension palette contributions,
 * and settings into the palette. Widget-scoped commands are filtered to widget
 * types present in the current layout, and execute through the contribution's
 * own source so widget handlers resolve exactly as they do for hotkeys.
 */

const SETTINGS_ENTRY_DEPS = {
  openSettingsSection: openWorkbenchSettings,
  patchPreferences: patchWorkbenchPreferences,
};

const STATIC_APP_ENTRIES: PaletteEntry[] = [
  {
    group: 'App',
    id: 'app.openSettings',
    keywords: 'preferences options',
    run: () => openWorkbenchSettings(),
    showInEmptyState: true,
    title: 'Open Settings',
  },
  {
    group: 'App',
    id: 'app.openHotkeySettings',
    keywords: 'hotkeys keybindings shortcuts',
    run: () => openWorkbenchSettings('hotkeys'),
    title: 'Keyboard Shortcuts',
  },
];

export const WorkbenchCommandPalette = () => {
  const isOpen = commandPaletteStore.useSelector((snapshot) => snapshot.isOpen);
  const extensions = useWorkbenchExtensions();
  const preferences = useWorkbenchPreferences();
  const projectId = useActiveProjectSelector((project) => project.id);
  const presentWidgetTypeIds = useActiveProjectSelector((project) =>
    [...new Set(Object.values(project.widgetInstances).map((instance) => instance.typeId))].sort()
  );
  const paletteStore = extensions.stores.palette;
  const paletteContributions = useSyncExternalStore(paletteStore.subscribe, paletteStore.list, paletteStore.list);

  const executeCommand = useCallback(
    (commandId: string) => {
      // Widget handlers register with a source; reuse it so scoped commands
      // resolve, falling back to the sourceless (global) registration.
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
        presentWidgetTypeIds: new Set(presentWidgetTypeIds),
      }),
      ...paletteContributions.map<PaletteEntry>((contribution) => ({
        group: 'Commands',
        id: `palette:${contribution.commandId}`,
        keywords: contribution.keywords?.join(' '),
        run: () => extensions.commands.executeForSource(contribution.commandId, contribution.source ?? null),
        title: contribution.title,
      })),
      ...STATIC_APP_ENTRIES,
      ...buildSettingsEntries(preferences, SETTINGS_ENTRY_DEPS),
    ],
    [executeCommand, extensions, paletteContributions, preferences, presentWidgetTypeIds]
  );

  const workbenchCommands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const providers = useMemo<PaletteSearchProvider[]>(() => {
    const { gallery, widgets } = workbenchCommands;
    const openWidget = (typeId: 'workflow' | 'gallery' | 'generate') =>
      openWidgetPlacement({
        getWidgetsForRegion,
        options:
          typeId === 'workflow'
            ? { preferredRegions: ['center'], requireCenterView: true }
            : { preferredRegions: typeId === 'gallery' ? ['center', 'right'] : ['left'] },
        typeId,
        widgets,
      });

    return [
      createWorkflowsProvider({ openWorkflowWidget: () => openWidget('workflow') }),
      createBoardsProvider({
        openGalleryWidget: () => openWidget('gallery'),
        selectBoard: (boardId) => gallery.selectBoard(boardId),
      }),
      createPromptHistoryProvider({
        getPromptHistory: () => queries.getSnapshot().activeProject.promptHistory,
        openGenerateWidget: () => openWidget('generate'),
        recallPrompt: (item) => {
          const project = queries.getSnapshot().activeProject;
          const patch: Record<string, unknown> = { positivePrompt: item.positivePrompt };

          if (item.negativePrompt) {
            patch.negativePrompt = item.negativePrompt;
            patch.negativePromptEnabled = true;
          }

          widgets.patchValues('generate', patch, project.id);
        },
      }),
    ];
  }, [queries, workbenchCommands]);

  return <CommandPaletteDialog entries={entries} isOpen={isOpen} providers={providers} onClose={closeCommandPalette} />;
};
