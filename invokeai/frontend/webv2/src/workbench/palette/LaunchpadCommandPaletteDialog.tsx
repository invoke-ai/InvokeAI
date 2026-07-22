import type { WorkbenchPreferences } from '@workbench/settings/contracts';

import { useCapabilities } from '@features/identity';
import { useNavigate, useSearch } from '@tanstack/react-router';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { PaletteEntry, SettingsEntryDeps } from './entries';

import { CommandPaletteDialog } from './CommandPaletteDialog';
import { buildOpenSettingsEntry, buildSettingsEntries } from './entries';

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
    const navEntry = ({ id, keywords, run }: { id: string; keywords?: string; run: () => void }): PaletteEntry => ({
      group: 'Navigation',
      groupLabel: t('commandPalette.groups.navigation'),
      id: `launchpad.${id}`,
      isPersistentRecent: true,
      keywords,
      run,
      showInEmptyState: true,
      title: t(`commandPalette.launchpad.${id}`),
    });
    const navigation: PaletteEntry[] = [
      navEntry({
        id: 'openEditor',
        keywords: 'workbench app project',
        run: () => void navigate({ search: search.project ? { project: search.project } : {}, to: '/app' }),
      }),
      navEntry({ id: 'goToProjects', run: () => void navigate({ to: '/projects' }) }),
      ...(canManageModels ? [navEntry({ id: 'goToModels', run: () => void navigate({ to: '/models' }) })] : []),
      ...(canManageNodes ? [navEntry({ id: 'goToNodes', run: () => void navigate({ to: '/nodes' }) })] : []),
      ...(canManageUsers ? [navEntry({ id: 'goToUsers', run: () => void navigate({ to: '/users' }) })] : []),
      buildOpenSettingsEntry(t, () => openWorkbenchSettings()),
    ];

    return [...navigation, ...buildSettingsEntries(preferences, settingsEntryDeps, t)];
  }, [canManageModels, canManageNodes, canManageUsers, navigate, preferences, search.project, settingsEntryDeps, t]);

  return <CommandPaletteDialog entries={entries} isOpen modifierKeyLabel={modifierKeyLabel} onClose={onClose} />;
};

export default LaunchpadCommandPaletteDialog;
