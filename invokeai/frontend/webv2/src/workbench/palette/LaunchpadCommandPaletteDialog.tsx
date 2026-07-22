import { useCapabilities } from '@features/identity';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { PaletteEntry, SettingsEntryDeps } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';
import { buildSettingsEntries } from './entries';

/** Launchpad-only palette adapter: navigation and settings, no editor providers. */
const LaunchpadCommandPaletteDialog = ({
  modifierKeyLabel,
  onClose,
  preferences,
  settingsEntryDeps,
}: {
  modifierKeyLabel: string;
  onClose: () => void;
  preferences: WorkbenchPreferences;
  settingsEntryDeps: SettingsEntryDeps;
}) => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const search = useSearch({ strict: false }) as { project?: string };
  const { canManageModels, canManageNodes, canManageUsers } = useCapabilities();

  const entries = useMemo<PaletteEntry[]>(() => {
    const navigation: PaletteEntry[] = [
      {
        group: 'Navigation',
        groupLabel: t('commandPalette.groups.navigation'),
        id: 'launchpad.openEditor',
        isPersistentRecent: true,
        keywords: 'workbench app project',
        run: () => void navigate({ search: search.project ? { project: search.project } : {}, to: '/app' }),
        showInEmptyState: true,
        title: t('commandPalette.launchpad.openEditor'),
      },
      {
        group: 'Navigation',
        groupLabel: t('commandPalette.groups.navigation'),
        id: 'launchpad.goToProjects',
        isPersistentRecent: true,
        run: () => void navigate({ to: '/projects' }),
        showInEmptyState: true,
        title: t('commandPalette.launchpad.goToProjects'),
      },
      ...(canManageModels
        ? [
            {
              group: 'Navigation',
              groupLabel: t('commandPalette.groups.navigation'),
              id: 'launchpad.goToModels',
              isPersistentRecent: true,
              run: () => void navigate({ to: '/models' }),
              showInEmptyState: true,
              title: t('commandPalette.launchpad.goToModels'),
            },
          ]
        : []),
      ...(canManageNodes
        ? [
            {
              group: 'Navigation',
              groupLabel: t('commandPalette.groups.navigation'),
              id: 'launchpad.goToNodes',
              isPersistentRecent: true,
              run: () => void navigate({ to: '/nodes' }),
              showInEmptyState: true,
              title: t('commandPalette.launchpad.goToNodes'),
            },
          ]
        : []),
      ...(canManageUsers
        ? [
            {
              group: 'Navigation',
              groupLabel: t('commandPalette.groups.navigation'),
              id: 'launchpad.goToUsers',
              isPersistentRecent: true,
              run: () => void navigate({ to: '/users' }),
              showInEmptyState: true,
              title: t('commandPalette.launchpad.goToUsers'),
            },
          ]
        : []),
      {
        group: 'App',
        groupLabel: t('commandPalette.groups.app'),
        id: 'app.openSettings',
        isPersistentRecent: true,
        keywords: 'preferences options',
        run: () => openWorkbenchSettings(),
        showInEmptyState: true,
        title: t('commandPalette.launchpad.openSettings'),
      },
    ];

    return [...navigation, ...buildSettingsEntries(preferences, settingsEntryDeps, t)];
  }, [canManageModels, canManageNodes, canManageUsers, navigate, preferences, search.project, settingsEntryDeps, t]);

  return <CommandPaletteDialog entries={entries} isOpen modifierKeyLabel={modifierKeyLabel} onClose={onClose} />;
};

export default LaunchpadCommandPaletteDialog;
import type { WorkbenchPreferences } from '@workbench/settings/contracts';
