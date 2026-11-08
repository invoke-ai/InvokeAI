import type { LogNamespace } from '@platform/logging/contracts';
import type { SettingFieldProps, SettingLayer } from '@platform/ui/settings/contracts';
import type { WorkbenchThemeId } from '@theme/themes';

import { Box, chakra, Checkbox, Flex, HStack, Icon, SimpleGrid, Stack, Text, useSlotRecipe } from '@chakra-ui/react';
import { useCapabilities } from '@features/identity';
import { INTERMEDIATES_SETTING_ID } from '@features/intermediates';
import { Button } from '@platform/ui/Button';
import { resolveSettingsText } from '@platform/ui/settings/contracts';
import { SettingLayerIndicator } from '@platform/ui/settings/SettingLayerIndicator';
import { themeCardRecipe } from '@theme/recipes';
import { previewSwatches, THEMES, type ThemeDefinition } from '@theme/system';
import { areLoggingPreferencesDefault, resetLoggingPreferences } from '@workbench/diagnostics/loggingPreferences';
import { useOptionalWorkbenchCommands } from '@workbench/WorkbenchContext';
import { BrushCleaningIcon, CheckIcon, DatabaseIcon, RotateCcwIcon } from 'lucide-react';
import { lazy, Suspense, useCallback, useMemo, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { AboutSettings } from './AboutSettings';
import { ClearSavedDataSettings } from './ClearSavedDataSettings';
import { GenerationDevicesSettings } from './GenerationDevicesSettings';
import { HotkeysSettingsSection } from './HotkeysSettingsSection';
import { ImageMapVocabularySettings } from './ImageMapVocabularySettings';
import { ImageStorageSettings } from './ImageStorageSettings';
import { MaxQueueHistorySettings } from './MaxQueueHistorySettings';
import {
  DEFAULT_PREFERENCES,
  DEVELOPER_LOG_NAMESPACES,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
} from './store';

const THEME_GRID_COLUMNS = { base: 2, md: 3 };
const DEVELOPER_GRID_COLUMNS = { base: 1, md: 2 };
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
  const [isDatabaseDialogMounted, setIsDatabaseDialogMounted] = useState(false);
  const [isDatabaseDialogOpen, setIsDatabaseDialogOpen] = useState(false);

  const resetLayout = useCallback(() => commands?.layout.reset(), [commands]);
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
            {t('settings.revertLayout')}
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
      </HStack>
      {canManageAppConfig ? (
        <Suspense fallback={null}>
          <LazyGalleryMaintenance />
        </Suspense>
      ) : null}
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
  const layer = useMemo<SettingLayer | undefined>(
    () =>
      isModified
        ? {
            onResetToBuiltIn: () =>
              void patchWorkbenchPreferences(
                field.id === 'themeId'
                  ? { themeId: DEFAULT_PREFERENCES.themeId }
                  : { developerLogNamespaces: DEFAULT_PREFERENCES.developerLogNamespaces }
              ),
            state: 'default',
          }
        : undefined,
    [field.id, isModified]
  );
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
    case 'clearSavedData':
      // A danger area of its own, apart from the everyday workspace actions; it carries its own heading.
      return (
        <ClearSavedDataSettings
          description={field.description ? resolveSettingsText(field.description, t) : ''}
          label={resolveSettingsText(field.label, t)}
        />
      );
    case 'about':
      editor = <AboutSettings />;
      break;
    case 'generationDevices':
      editor = <GenerationDevicesSettings />;
      break;
    case 'maxQueueHistory':
      editor = <MaxQueueHistorySettings />;
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
    case 'imageStorage':
      editor = <ImageStorageSettings />;
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
          {layer ? <SettingLayerIndicator label={resolveSettingsText(field.label, t)} layer={layer} /> : null}
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
