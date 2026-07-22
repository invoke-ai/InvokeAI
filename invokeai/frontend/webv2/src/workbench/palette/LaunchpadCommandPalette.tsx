import { useCapabilities } from '@features/identity';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { patchWorkbenchPreferences, useWorkbenchPreferences } from '@workbench/settings/store';
import { useEffect, useMemo } from 'react';
import { tinykeys } from 'tinykeys';

import type { PaletteEntry } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';
import { buildSettingsEntries } from './entries';
import { closeCommandPalette, commandPaletteStore, toggleCommandPalette } from './paletteStore';

/**
 * Launchpad host: the editor's hotkey runtime is not mounted here, so this
 * host binds mod+K itself. The source set is deliberately reduced — section
 * navigation and settings only; editor commands need an editor mount. A mod+K
 * that works "sometimes" trains users not to trust it.
 */

const SETTINGS_ENTRY_DEPS = {
  openSettingsSection: openWorkbenchSettings,
  patchPreferences: patchWorkbenchPreferences,
};

export const LaunchpadCommandPalette = () => {
  const isOpen = commandPaletteStore.useSelector((snapshot) => snapshot.isOpen);
  const navigate = useNavigate();
  const search = useSearch({ strict: false }) as { project?: string };
  const preferences = useWorkbenchPreferences();
  const { canManageModels, canManageNodes, canManageUsers } = useCapabilities();

  useEffect(
    () =>
      tinykeys(window, {
        // preventDefault beats Firefox's Ctrl+K address-bar search; macOS has
        // no browser binding on Cmd+K.
        '$mod+KeyK': (event) => {
          event.preventDefault();
          toggleCommandPalette();
        },
      }),
    []
  );

  const entries = useMemo<PaletteEntry[]>(() => {
    const navigation: PaletteEntry[] = [
      {
        group: 'Navigation',
        id: 'launchpad.openEditor',
        keywords: 'workbench app project',
        run: () => void navigate({ search: search.project ? { project: search.project } : {}, to: '/app' }),
        showInEmptyState: true,
        title: 'Open Editor',
      },
      {
        group: 'Navigation',
        id: 'launchpad.goToProjects',
        run: () => void navigate({ to: '/projects' }),
        showInEmptyState: true,
        title: 'Go to Projects',
      },
      ...(canManageModels
        ? [
            {
              group: 'Navigation',
              id: 'launchpad.goToModels',
              run: () => void navigate({ to: '/models' }),
              showInEmptyState: true,
              title: 'Go to Models',
            },
          ]
        : []),
      ...(canManageNodes
        ? [
            {
              group: 'Navigation',
              id: 'launchpad.goToNodes',
              run: () => void navigate({ to: '/nodes' }),
              showInEmptyState: true,
              title: 'Go to Nodes',
            },
          ]
        : []),
      ...(canManageUsers
        ? [
            {
              group: 'Navigation',
              id: 'launchpad.goToUsers',
              run: () => void navigate({ to: '/users' }),
              showInEmptyState: true,
              title: 'Go to Users',
            },
          ]
        : []),
      {
        group: 'App',
        id: 'app.openSettings',
        keywords: 'preferences options',
        run: () => openWorkbenchSettings(),
        showInEmptyState: true,
        title: 'Open Settings',
      },
    ];

    return [...navigation, ...buildSettingsEntries(preferences, SETTINGS_ENTRY_DEPS)];
  }, [canManageModels, canManageNodes, canManageUsers, navigate, preferences, search.project]);

  return <CommandPaletteDialog entries={entries} isOpen={isOpen} onClose={closeCommandPalette} />;
};
