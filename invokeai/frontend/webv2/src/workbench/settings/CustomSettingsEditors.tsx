import type { LogNamespace } from '@platform/logging/contracts';
import type { SettingFieldProps } from '@platform/ui/settings/contracts';
import type { WorkbenchThemeId } from '@theme/themes';

import { Box, chakra, Checkbox, Flex, HStack, Icon, SimpleGrid, Stack, Text, useSlotRecipe } from '@chakra-ui/react';
import { useCapabilities } from '@features/identity';
import { INTERMEDIATES_SETTING_ID } from '@features/intermediates';
import { Button, ConfirmDialog } from '@platform/ui';
import { resolveSettingsText } from '@platform/ui/settings/contracts';
import { ModifiedSettingIndicator } from '@platform/ui/settings/ModifiedSettingIndicator';
import { themeCardRecipe } from '@theme/recipes';
import { previewSwatches, THEMES, type ThemeDefinition } from '@theme/system';
import { areLoggingPreferencesDefault, resetLoggingPreferences } from '@workbench/diagnostics/loggingPreferences';
import { clearAllWorkbenchData } from '@workbench/projects/syncedPersistence';
import { useOptionalWorkbenchCommands, useOptionalWorkbenchPersistenceService } from '@workbench/WorkbenchContext';
import { BrushCleaningIcon, CheckIcon, DatabaseIcon, RotateCcwIcon, Trash2Icon } from 'lucide-react';
import { lazy, Suspense, useCallback, useMemo, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { AboutSettings } from './AboutSettings';
import { clearWorkspaceData, rememberWorkspaceClearFailure } from './clearWorkspaceData';
import { GenerationDevicesSettings } from './GenerationDevicesSettings';
import { HotkeysSettingsSection } from './HotkeysSettingsSection';
import { ImageMapVocabularySettings } from './ImageMapVocabularySettings';
import {
  clearWorkbenchSettings,
  DEFAULT_PREFERENCES,
  DEVELOPER_LOG_NAMESPACES,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
  useWorkbenchSettingsSelector,
} from './store';

const THEME_GRID_COLUMNS = { base: 2, md: 3 };
const DEVELOPER_GRID_COLUMNS = { base: 1, md: 2 };
const DANGER_BUTTON_HOVER_STYLES = { bg: 'fg.error', color: 'bg.subtle' };
const LazyDatabaseMaintenanceDialog = lazy(() =>
  import('./DatabaseMaintenanceDialog').then((module) => ({ default: module.DatabaseMaintenanceDialog }))
);
const LazyGalleryMaintenance = lazy(() =>
  import('./GalleryMaintenance').then((module) => ({ default: module.GalleryMaintenance }))
);

export const ThemeSettings = () => {
  const themeId = useWorkbenchPreferenceSelector((preferences) => preferences.themeId);
  const selectTheme = useCallback((nextThemeId: WorkbenchThemeId) => {
    void patchWorkbenchPreferences({ themeId: nextThemeId });
  }, []);

  return (
    <SimpleGrid columns={THEME_GRID_COLUMNS} gap="3">
      {THEMES.map((theme) => (
        <ThemeCard key={theme.id} selected={theme.id === themeId} theme={theme} onSelect={selectTheme} />
      ))}
    </SimpleGrid>
  );
};

const ThemeCard = ({
  onSelect,
  selected,
  theme,
}: {
  onSelect: (themeId: WorkbenchThemeId) => void;
  selected: boolean;
  theme: ThemeDefinition;
}) => {
  const recipe = useSlotRecipe({ recipe: themeCardRecipe });
  const styles = recipe({ selected });
  const [surface, control, brandColor, accentColor] = previewSwatches(theme);
  const handleSelect = useCallback(() => onSelect(theme.id), [onSelect, theme.id]);

  return (
    <chakra.button type="button" aria-pressed={selected} css={styles.root} onClick={handleSelect}>
      <Flex css={styles.preview}>
        <Box css={styles.swatch} bg={surface} />
        <Box css={styles.swatch} bg={control} />
        <Box css={styles.swatch} bg={brandColor} />
        <Box css={styles.swatch} bg={accentColor} />
      </Flex>
      <Box css={styles.body}>
        <HStack justify="space-between" w="full">
          <Text css={styles.name}>{theme.label}</Text>
          <Box css={styles.indicator}>
            <Icon as={CheckIcon} boxSize="3" />
          </Box>
        </HStack>
        <Text css={styles.description}>{theme.description}</Text>
      </Box>
    </chakra.button>
  );
};

export const DeveloperNamespacesSettings = () => {
  const developerLogNamespaces = useWorkbenchPreferenceSelector((preferences) => preferences.developerLogNamespaces);
  const enabledNamespaces = useMemo(() => new Set(developerLogNamespaces), [developerLogNamespaces]);
  const toggleNamespace = useCallback(
    (namespace: LogNamespace, checked: boolean) => {
      const next = checked
        ? [...developerLogNamespaces, namespace]
        : developerLogNamespaces.filter((candidate) => candidate !== namespace);

      void patchWorkbenchPreferences({
        developerLogNamespaces: DEVELOPER_LOG_NAMESPACES.filter((candidate) => next.includes(candidate)),
      });
    },
    [developerLogNamespaces]
  );

  return (
    <SimpleGrid columns={DEVELOPER_GRID_COLUMNS} gap="2">
      {DEVELOPER_LOG_NAMESPACES.map((namespace) => (
        <DeveloperNamespaceCheckbox
          key={namespace}
          checked={enabledNamespaces.has(namespace)}
          namespace={namespace}
          toggleNamespace={toggleNamespace}
        />
      ))}
    </SimpleGrid>
  );
};

const DeveloperNamespaceCheckbox = ({
  checked,
  namespace,
  toggleNamespace,
}: {
  checked: boolean;
  namespace: LogNamespace;
  toggleNamespace: (namespace: LogNamespace, checked: boolean) => void;
}) => {
  const handleCheckedChange = useCallback(
    (event: { checked: boolean | 'indeterminate' }) => toggleNamespace(namespace, event.checked === true),
    [namespace, toggleNamespace]
  );

  return (
    <Checkbox.Root checked={checked} onCheckedChange={handleCheckedChange}>
      <Checkbox.HiddenInput />
      <Checkbox.Control />
      <Checkbox.Label color="fg.muted" fontSize="md">
        {formatSettingLabel(namespace)}
      </Checkbox.Label>
    </Checkbox.Root>
  );
};

export const LoggingResetSettings = () => {
  const { t } = useTranslation();
  const isDefault = useWorkbenchPreferenceSelector(areLoggingPreferencesDefault);
  const reset = useCallback(() => void resetLoggingPreferences(), []);

  return (
    <HStack gap="3">
      <Button disabled={isDefault} size="lg" variant="outline" onClick={reset}>
        <RotateCcwIcon />
        {t('settings.catalog.resetLoggingDefaults')}
      </Button>
      {isDefault ? (
        <Text color="fg.muted" fontSize="md">
          {t('settings.catalog.loggingDefaultsActive')}
        </Text>
      ) : null}
    </HStack>
  );
};

export const WorkspaceSettings = ({ onReveal }: Pick<SettingFieldProps, 'onReveal'>) => {
  const { t } = useTranslation();
  const { canManageAppConfig } = useCapabilities();
  const commands = useOptionalWorkbenchCommands();
  const mountedPersistence = useOptionalWorkbenchPersistenceService();
  const scope = useWorkbenchSettingsSelector((snapshot) => snapshot.scope);
  const [isClearConfirmOpen, setIsClearConfirmOpen] = useState(false);
  const [isDatabaseDialogMounted, setIsDatabaseDialogMounted] = useState(false);
  const [isDatabaseDialogOpen, setIsDatabaseDialogOpen] = useState(false);

  const clearSavedData = useCallback(async () => {
    const failures = await clearWorkspaceData(
      () => (mountedPersistence ? mountedPersistence.clearWorkbench() : clearAllWorkbenchData()),
      clearWorkbenchSettings
    );
    if (failures.length === 0) {
      window.location.reload();
      return;
    }
    const message = failures.includes('projects')
      ? 'Saved project data could not be fully cleared. Reloading to restore a consistent workspace.'
      : 'Projects were cleared, but some local settings could not be. Reloading the workspace.';
    rememberWorkspaceClearFailure(message, window.sessionStorage);
    commands?.notifications.reportError({ area: 'workspace-clear', message, namespace: 'system' });
    window.location.reload();
  }, [commands, mountedPersistence]);
  const resetLayout = useCallback(() => commands?.layout.reset(), [commands]);
  const openClearConfirm = useCallback(() => setIsClearConfirmOpen(true), []);
  const closeClearConfirm = useCallback(() => setIsClearConfirmOpen(false), []);
  const openDatabaseConfirmation = useCallback(() => {
    setIsDatabaseDialogMounted(true);
    setIsDatabaseDialogOpen(true);
  }, []);
  const closeDatabaseConfirmation = useCallback(() => setIsDatabaseDialogOpen(false), []);
  const openIntermediates = useCallback(() => onReveal?.('intermediates', INTERMEDIATES_SETTING_ID), [onReveal]);

  return (
    <Stack gap="3">
      <HStack gap="2" wrap="wrap">
        {commands ? (
          <Button size="lg" variant="outline" onClick={resetLayout}>
            <RotateCcwIcon />
            Reset layout
          </Button>
        ) : null}
        {onReveal ? (
          <Button size="lg" variant="outline" onClick={openIntermediates}>
            <BrushCleaningIcon />
            {t('settings.catalog.manageIntermediates')}
          </Button>
        ) : null}
        {canManageAppConfig ? (
          <Button size="lg" variant="outline" onClick={openDatabaseConfirmation}>
            <DatabaseIcon />
            {t('settings.databaseMaintenance.compactDatabase')}
          </Button>
        ) : null}
        <Button
          borderColor="border.emphasized"
          color="fg.error"
          size="lg"
          variant="outline"
          _hover={DANGER_BUTTON_HOVER_STYLES}
          onClick={openClearConfirm}
        >
          <Trash2Icon />
          Clear saved data…
        </Button>
      </HStack>
      {canManageAppConfig ? (
        <Suspense fallback={null}>
          <LazyGalleryMaintenance />
        </Suspense>
      ) : null}
      <ConfirmDialog
        body={
          scope === 'user'
            ? 'This permanently deletes all projects and settings for your account on this server. It cannot be undone.'
            : 'This permanently deletes all projects and settings for this install. It cannot be undone.'
        }
        confirmLabel="Delete everything"
        isOpen={isClearConfirmOpen}
        title="Clear saved data?"
        onClose={closeClearConfirm}
        onConfirm={clearSavedData}
      />
      {canManageAppConfig && isDatabaseDialogMounted ? (
        <Suspense fallback={null}>
          <LazyDatabaseMaintenanceDialog isOpen={isDatabaseDialogOpen} onClose={closeDatabaseConfirmation} />
        </Suspense>
      ) : null}
    </Stack>
  );
};

const formatSettingLabel = (value: string): string =>
  value
    .split('-')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');

const CustomSettingField = ({ field, onReveal }: SettingFieldProps) => {
  const { t } = useTranslation();
  const isModified = useWorkbenchPreferenceSelector((preferences) => {
    if (field.id === 'themeId') {
      return preferences.themeId !== DEFAULT_PREFERENCES.themeId;
    }
    if (field.id === 'developerLogNamespaces') {
      return DEVELOPER_LOG_NAMESPACES.some(
        (namespace) =>
          preferences.developerLogNamespaces.includes(namespace) !==
          DEFAULT_PREFERENCES.developerLogNamespaces.includes(namespace)
      );
    }
    return false;
  });
  let editor: ReactNode;

  switch (field.id) {
    case 'themeId':
      editor = <ThemeSettings />;
      break;
    case 'developerLogNamespaces':
      editor = <DeveloperNamespacesSettings />;
      break;
    case 'developerLoggingReset':
      editor = <LoggingResetSettings />;
      break;
    case 'workspaceActions':
      editor = <WorkspaceSettings onReveal={onReveal} />;
      break;
    case 'about':
      editor = <AboutSettings />;
      break;
    case 'generationDevices':
      editor = <GenerationDevicesSettings />;
      break;
    case 'hotkeys':
      // The hotkey editor owns its heading and fills the dialog's allocated height.
      return (
        <Box display="flex" flex="1" flexDirection="column" minH="0">
          <HotkeysSettingsSection />
        </Box>
      );
    case 'imageMapVocabulary':
      editor = <ImageMapVocabularySettings />;
      break;
    default:
      return null;
  }

  return (
    <Stack gap="3" w="full">
      <Stack gap="1">
        <HStack gap="2">
          <Text color="fg" fontSize="lg" fontWeight="500">
            {resolveSettingsText(field.label, t)}
          </Text>
          {isModified ? <ModifiedSettingIndicator label={resolveSettingsText(field.label, t)} /> : null}
        </HStack>
        {field.description ? (
          <Text color="fg.muted" fontSize="md">
            {resolveSettingsText(field.description, t)}
          </Text>
        ) : null}
      </Stack>
      {editor}
    </Stack>
  );
};

export default CustomSettingField;
