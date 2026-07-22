import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import { useCapabilities } from '@features/identity';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { useMemo } from 'react';

import type { PaletteEntry, SettingsEntryDeps } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';
import { buildSettingsEntries } from './entries';

/** Launchpad-only palette adapter: navigation and settings, no editor providers. */
const LaunchpadCommandPaletteDialog = ({
  onClose,
  modifierKeyLabel,
  preferences,
  settingsEntryDeps,
}: {
  onClose: () => void;
  modifierKeyLabel: string;
  preferences: WorkbenchPreferences;
  settingsEntryDeps: SettingsEntryDeps;
}) => {
  const navigate = useNavigate();
  const search = useSearch({ strict: false }) as { project?: string };
  const { canManageModels, canManageNodes, canManageUsers } = useCapabilities();

  const entries = useMemo<PaletteEntry[]>(() => {
    const navigation: PaletteEntry[] = [
      {
        group: 'Navigation',
        id: 'launchpad.openEditor',
        isPersistentRecent: true,
        keywords: 'workbench app project',
        run: () => void navigate({ search: search.project ? { project: search.project } : {}, to: '/app' }),
        showInEmptyState: true,
        title: 'Open Editor',
      },
      {
        group: 'Navigation',
        id: 'launchpad.goToProjects',
        isPersistentRecent: true,
        run: () => void navigate({ to: '/projects' }),
        showInEmptyState: true,
        title: 'Go to Projects',
      },
      ...(canManageModels
        ? [
            {
              group: 'Navigation',
              id: 'launchpad.goToModels',
              isPersistentRecent: true,
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
              isPersistentRecent: true,
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
              isPersistentRecent: true,
              run: () => void navigate({ to: '/users' }),
              showInEmptyState: true,
              title: 'Go to Users',
            },
          ]
        : []),
      {
        group: 'App',
        id: 'app.openSettings',
        isPersistentRecent: true,
        keywords: 'preferences options',
        run: () => openWorkbenchSettings(),
        showInEmptyState: true,
        title: 'Open Settings',
      },
    ];

    return [...navigation, ...buildSettingsEntries(preferences, settingsEntryDeps)];
  }, [canManageModels, canManageNodes, canManageUsers, navigate, preferences, search.project, settingsEntryDeps]);

  return <CommandPaletteDialog entries={entries} isOpen modifierKeyLabel={modifierKeyLabel} onClose={onClose} />;
};

export default LaunchpadCommandPaletteDialog;
