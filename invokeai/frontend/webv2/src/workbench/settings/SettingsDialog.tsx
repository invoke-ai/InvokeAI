import type {
  DeveloperLogLevel,
  DeveloperLogNamespace,
  ProjectSettings,
  Project,
  SettingsSectionId,
  WorkbenchLanguage,
  WorkbenchPreferences,
  WorkbenchThemeId,
} from '@workbench/types';

import {
  Box,
  chakra,
  Checkbox,
  Dialog,
  Flex,
  HStack,
  Icon,
  NativeSelect,
  Portal,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  Field,
  useSlotRecipe,
} from '@chakra-ui/react';
import { themeCardRecipe } from '@theme/recipes';
import { previewSwatches, THEMES, type ThemeDefinition } from '@theme/system';
import { Button, CloseButton, IconButton, ConfirmDialog, Tabs, Tooltip } from '@workbench/components/ui';
import { registerHotkeyModalLayer } from '@workbench/hotkeys';
import { syncedWorkbenchPersistence } from '@workbench/projects/syncedPersistence';
import {
  shallowEqual,
  useOptionalWorkbenchDispatch,
  useOptionalWorkbenchSelector,
  useOptionalWorkbenchStore,
} from '@workbench/WorkbenchContext';
import {
  CheckIcon,
  Code2Icon,
  DatabaseIcon,
  FolderIcon,
  KeyboardIcon,
  ListOrderedIcon,
  PaletteIcon,
  RotateCcwIcon,
  SettingsIcon,
  SlidersHorizontalIcon,
  Trash2Icon,
  WorkflowIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, useEffect, useId, useMemo, useState, type ReactNode } from 'react';

import { ClearIntermediatesSetting } from './ClearIntermediatesSetting';
import { HotkeysSettingsSection } from './HotkeysSettingsSection';
import {
  closeWorkbenchSettings,
  openWorkbenchSettings,
  setWorkbenchSettingsSection,
  settingsDialogStore,
} from './settingsDialogStore';
import {
  clearWorkbenchSettings,
  DEVELOPER_LOG_LEVELS,
  DEVELOPER_LOG_NAMESPACES,
  patchWorkbenchPreferences,
  useWorkbenchPreferences,
  useWorkbenchSettings,
  WORKBENCH_LANGUAGES,
} from './store';

interface SettingsTabDefinition {
  value: SettingsSectionId;
  label: string;
  icon: LucideIcon;
  condition?: boolean;
  children: ReactNode;
}

const updatePreferences = (patch: Partial<WorkbenchPreferences>): void => {
  void patchWorkbenchPreferences(patch);
};

const SETTINGS_TAB_LIST_WIDTH = { base: '40', md: '52' };
const SETTINGS_CONTENT_PADDING = { base: '4', md: '5' };
const THEME_GRID_COLUMNS = { base: 2, md: 3 };
const DEVELOPER_GRID_COLUMNS = { base: 1, md: 2 };
const DANGER_BUTTON_HOVER_STYLES = { bg: 'fg.error', color: 'bg.subtle' };
const SWITCH_CHECKED_STYLES = { bg: 'accent.solid' };
const FIELD_ALIGN_ITEMS = { base: 'stretch', md: 'center' };
const FIELD_FLEX_DIRECTION = { base: 'column', md: 'row' };
const SELECT_MAX_WIDTH = { base: 'full', md: '56' };

/**
 * The top bar's settings entry point. It also hosts the dialog itself, driven
 * by `settingsDialogStore` so any surface can deep-link into a section via
 * `openWorkbenchSettings('workflow')`.
 */
export const SettingsButton = () => {
  const isOpen = settingsDialogStore.useSelector((snapshot) => snapshot.isOpen);
  const handleOpen = useCallback(() => openWorkbenchSettings(), []);

  return (
    <>
      <Tooltip content="Settings">
        <IconButton aria-label="Settings" size="sm" variant="ghost" onClick={handleOpen}>
          <SettingsIcon />
        </IconButton>
      </Tooltip>
      <SettingsDialog isOpen={isOpen} onClose={closeWorkbenchSettings} />
    </>
  );
};

export const SettingsDialog = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  useEffect(() => {
    if (!isOpen) {
      return;
    }

    return registerHotkeyModalLayer('settings');
  }, [isOpen]);

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Dialog.Root
      closeOnInteractOutside={false}
      lazyMount
      open={isOpen}
      placement="center"
      scrollBehavior="inside"
      size="xl"
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <SettingsDialogContent onClose={onClose} />
    </Dialog.Root>
  );
};

const SettingsDialogContent = ({ onClose }: { onClose: () => void }) => {
  const { error, scope, status } = useWorkbenchSettings();
  const handlePositionerClick = useCallback(
    (event: { target: EventTarget; currentTarget: EventTarget }) => {
      if (event.target === event.currentTarget) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Portal>
      <Dialog.Backdrop pointerEvents="auto" />
      <Dialog.Positioner pointerEvents="auto" onClick={handlePositionerClick}>
        <Dialog.Content h="min(46rem, calc(100dvh - 4rem))" maxW="4xl">
          <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
            <Flex alignItems="start" gap="2">
              <Icon as={SettingsIcon} boxSize="5" />
              <Stack gap="1">
                <Dialog.Title fontSize="md" fontWeight="700" mt="0.5" lineHeight={1}>
                  Settings
                </Dialog.Title>
                <Text color="fg.subtle" fontSize="xs">
                  {scope === 'user'
                    ? 'Preferences for the signed-in user. Project settings apply only to the active project.'
                    : 'Global preferences for this single-user install. Project settings apply only to the active project.'}
                </Text>
                {status === 'error' && error ? (
                  <Text color="fg.error" fontSize="2xs">
                    {error}
                  </Text>
                ) : null}
              </Stack>
            </Flex>
          </Dialog.Header>
          <Dialog.Body minH="0" p="0">
            <SettingsTabs />
          </Dialog.Body>
          <Dialog.CloseTrigger asChild>
            <CloseButton size="sm" />
          </Dialog.CloseTrigger>
        </Dialog.Content>
      </Dialog.Positioner>
    </Portal>
  );
};

const SettingsTabs = () => {
  const hasWorkbench = useOptionalWorkbenchStore() !== null;
  const settingsTabs: SettingsTabDefinition[] = useMemo(
    () => [
      {
        children: <AppearanceSection />,
        icon: PaletteIcon,
        label: 'Appearance',
        value: 'appearance',
      },
      {
        children: <BehaviorSection />,
        icon: SlidersHorizontalIcon,
        label: 'Behavior',
        value: 'behavior',
      },
      {
        children: <HotkeysSettingsSection />,
        icon: KeyboardIcon,
        label: 'Hotkeys',
        value: 'hotkeys',
      },
      {
        children: <ProjectSection />,
        condition: hasWorkbench,
        icon: FolderIcon,
        label: 'Project',
        value: 'project',
      },
      {
        children: <QueueSection />,
        icon: ListOrderedIcon,
        label: 'Queue',
        value: 'queue',
      },
      {
        children: <WorkflowSection />,
        icon: WorkflowIcon,
        label: 'Workflow',
        value: 'workflow',
      },
      {
        children: <DeveloperSection />,
        icon: Code2Icon,
        label: 'Developer',
        value: 'developer',
      },
      {
        children: <WorkspaceSection />,
        icon: DatabaseIcon,
        label: 'Workspace',
        value: 'workspace',
      },
    ],
    [hasWorkbench]
  );
  const tabs = settingsTabs.filter((tab) => tab.condition !== false);
  const sectionId = settingsDialogStore.useSelector((snapshot) => snapshot.sectionId);
  const activeSectionId = tabs.some((tab) => tab.value === sectionId) ? sectionId : 'appearance';
  const handleValueChange = useCallback((event: { value: string }) => {
    setWorkbenchSettingsSection(event.value as SettingsSectionId);
  }, []);

  return (
    <Tabs.Root
      display="flex"
      h="full"
      minH="0"
      orientation="vertical"
      value={activeSectionId}
      variant="subtle"
      onValueChange={handleValueChange}
    >
      <Tabs.List
        alignItems="stretch"
        bg="bg"
        borderColor="border.subtle"
        borderRightWidth="1px"
        flexShrink={0}
        p="2"
        w={SETTINGS_TAB_LIST_WIDTH}
      >
        {tabs.map((tab) => (
          <SettingsTabTrigger key={tab.value} icon={tab.icon} label={tab.label} value={tab.value} />
        ))}
      </Tabs.List>
      <Box flex="1" minW="0" overflowY="auto" p={SETTINGS_CONTENT_PADDING}>
        {tabs.map((tab) => (
          <Tabs.Content key={tab.value} m="0" p="0" value={tab.value}>
            {tab.children}
          </Tabs.Content>
        ))}
      </Box>
    </Tabs.Root>
  );
};

const SettingsTabTrigger = ({ icon, label, value }: { icon: LucideIcon; label: string; value: string }) => (
  <Tabs.Trigger justifyContent="flex-start" textAlign="start" value={value} w="full">
    <Icon as={icon} boxSize="3.5" flexShrink={0} />
    <Text truncate>{label}</Text>
  </Tabs.Trigger>
);

const SettingsSection = ({
  children,
  description,
  title,
}: {
  children: ReactNode;
  description: string;
  title: string;
}) => (
  <Stack gap="3">
    <Stack gap="0.5">
      <Text color="fg" fontSize="sm" fontWeight="600">
        {title}
      </Text>
      <Text color="fg.subtle" fontSize="xs">
        {description}
      </Text>
    </Stack>
    {children}
  </Stack>
);

const AppearanceSection = () => {
  const { language, reduceMotion, showFocusRegionHighlight, themeId } = useWorkbenchPreferences();

  const selectTheme = useCallback((nextThemeId: WorkbenchThemeId) => {
    updatePreferences({ themeId: nextThemeId });
  }, []);
  const updateLanguage = useCallback((value: string) => {
    updatePreferences({ language: value as WorkbenchLanguage });
  }, []);
  const updateReduceMotion = useCallback((checked: boolean) => {
    updatePreferences({ reduceMotion: checked });
  }, []);
  const updateShowFocusRegionHighlight = useCallback((checked: boolean) => {
    updatePreferences({ showFocusRegionHighlight: checked });
  }, []);

  return (
    <SettingsSection description="Choose a theme, language, and motion behavior." title="Appearance">
      <SimpleGrid columns={THEME_GRID_COLUMNS} gap="3">
        {THEMES.map((theme) => (
          <ThemeCard key={theme.id} selected={theme.id === themeId} theme={theme} onSelect={selectTheme} />
        ))}
      </SimpleGrid>
      <SettingSelect
        comingSoon
        description="Interface language. More i18n coverage will arrive soon."
        label="Language"
        value={language}
        onChange={updateLanguage}
      >
        {WORKBENCH_LANGUAGES.map((value) => (
          <option key={value} value={value}>
            {value}
          </option>
        ))}
      </SettingSelect>
      <SettingToggle
        checked={reduceMotion}
        description="Disable transitions and animations across the workbench."
        label="Reduce motion"
        onChange={updateReduceMotion}
      />
      <SettingToggle
        checked={showFocusRegionHighlight}
        description="Draw a subtle outline around the active workbench region."
        label="Highlight focused region"
        onChange={updateShowFocusRegionHighlight}
      />
    </SettingsSection>
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

const BehaviorSection = () => {
  const { confirmImageDeletion, enableInformationalPopovers, enableModelDescriptions } = useWorkbenchPreferences();
  const updateConfirmImageDeletion = useCallback((checked: boolean) => {
    updatePreferences({ confirmImageDeletion: checked });
  }, []);
  const updateEnableInformationalPopovers = useCallback((checked: boolean) => {
    updatePreferences({ enableInformationalPopovers: checked });
  }, []);
  const updateEnableModelDescriptions = useCallback((checked: boolean) => {
    updatePreferences({ enableModelDescriptions: checked });
  }, []);

  return (
    <SettingsSection description="Safety checks and user-assistance behavior." title="Behavior">
      <SettingToggle
        checked={confirmImageDeletion}
        description="Ask for confirmation before permanently deleting images."
        label="Confirm image deletion"
        onChange={updateConfirmImageDeletion}
      />
      <SettingToggle
        checked={enableInformationalPopovers}
        comingSoon
        description="Show educational popovers on controls that have extra guidance."
        label="Enable informational popovers"
        onChange={updateEnableInformationalPopovers}
      />
      <SettingToggle
        checked={enableModelDescriptions}
        description="Include model descriptions in model dropdowns where available."
        label="Enable model descriptions in dropdowns"
        onChange={updateEnableModelDescriptions}
      />
    </SettingsSection>
  );
};

const ProjectSection = () => {
  const activeProject = useOptionalWorkbenchSelector<Pick<Project, 'name' | 'settings'> | null>(
    (snapshot) => ({ name: snapshot.activeProject.name, settings: snapshot.activeProject.settings }),
    null,
    shallowEqual
  );
  const dispatch = useOptionalWorkbenchDispatch();
  const updateProjectSettings = useCallback(
    (patch: Partial<ProjectSettings>) => {
      dispatch?.({ settings: patch, type: 'setActiveProjectSettings' });
    },
    [dispatch]
  );
  const updateUseCpuNoise = useCallback(
    (checked: boolean) => {
      updateProjectSettings({ useCpuNoise: checked });
    },
    [updateProjectSettings]
  );
  const updateShowProgressDetails = useCallback(
    (checked: boolean) => {
      updateProjectSettings({ showProgressDetails: checked });
    },
    [updateProjectSettings]
  );
  const updateAntialiasProgressImages = useCallback(
    (checked: boolean) => {
      updateProjectSettings({ antialiasProgressImages: checked });
    },
    [updateProjectSettings]
  );
  const updateShowProgressImagesInViewer = useCallback(
    (checked: boolean) => {
      updateProjectSettings({ showProgressImagesInViewer: checked });
    },
    [updateProjectSettings]
  );
  const updatePreferNumericAttentionStyle = useCallback(
    (checked: boolean) => {
      updateProjectSettings({ preferNumericAttentionStyle: checked });
    },
    [updateProjectSettings]
  );
  const updateShowPromptSyntaxHighlighting = useCallback(
    (checked: boolean) => {
      updateProjectSettings({ showPromptSyntaxHighlighting: checked });
    },
    [updateProjectSettings]
  );

  if (!activeProject || !dispatch) {
    return null;
  }

  const settings = activeProject.settings;

  return (
    <SettingsSection
      description={`Settings saved with ${activeProject.name}. Future project-only settings can live here without becoming user preferences.`}
      title="Project"
    >
      <SettingToggle
        checked={settings.useCpuNoise}
        description="Use CPU noise generation for deterministic legacy-compatible outputs."
        label="Use CPU noise"
        onChange={updateUseCpuNoise}
      />
      <SettingToggle
        checked={settings.showProgressDetails}
        comingSoon
        description="Show detailed invocation progress when the backend reports it."
        label="Show progress details"
        onChange={updateShowProgressDetails}
      />
      <SettingToggle
        checked={settings.antialiasProgressImages}
        description="Smooth progress previews instead of rendering them pixelated."
        label="Antialias progress images"
        onChange={updateAntialiasProgressImages}
      />
      <SettingToggle
        checked={settings.showProgressImagesInViewer}
        description="Show progress images in the viewer when an image is still generating."
        label="Show progress images in viewer"
        onChange={updateShowProgressImagesInViewer}
      />
      <SettingToggle
        checked={settings.preferNumericAttentionStyle}
        description="Prefer numeric prompt attention syntax when controls insert attention weights."
        label="Prefer numeric attention style"
        onChange={updatePreferNumericAttentionStyle}
      />
      <SettingToggle
        checked={settings.showPromptSyntaxHighlighting}
        description="Experimental. Color prompt syntax in prompt fields without changing the prompt text or validation behavior."
        label="Highlight prompt syntax (experimental)"
        onChange={updateShowPromptSyntaxHighlighting}
      />
    </SettingsSection>
  );
};

const WorkflowSection = () => {
  const { workflowEdgeStyle, workflowShowMinimap, workflowSnapToGrid, workflowValidateConnections } =
    useWorkbenchPreferences();
  const updateWorkflowEdgeStyle = useCallback((value: string) => {
    updatePreferences({ workflowEdgeStyle: value === 'square' ? 'square' : 'curved' });
  }, []);
  const updateWorkflowSnapToGrid = useCallback((checked: boolean) => {
    updatePreferences({ workflowSnapToGrid: checked });
  }, []);
  const updateWorkflowShowMinimap = useCallback((checked: boolean) => {
    updatePreferences({ workflowShowMinimap: checked });
  }, []);
  const updateWorkflowValidateConnections = useCallback((checked: boolean) => {
    updatePreferences({ workflowValidateConnections: checked });
  }, []);

  return (
    <SettingsSection description="Editing behavior for the project graph workflow editor." title="Workflow">
      <SettingSelect
        description="How connections between nodes are drawn in the editor."
        label="Connection style"
        value={workflowEdgeStyle}
        onChange={updateWorkflowEdgeStyle}
      >
        <option value="curved">Curved</option>
        <option value="square">Square</option>
      </SettingSelect>
      <SettingToggle
        checked={workflowSnapToGrid}
        description="Snap nodes to the grid while dragging. When off, hold Ctrl to snap temporarily."
        label="Always snap to grid"
        onChange={updateWorkflowSnapToGrid}
      />
      <SettingToggle
        checked={workflowShowMinimap}
        description="Show the minimap overview in the corner of the workflow editor."
        label="Show minimap"
        onChange={updateWorkflowShowMinimap}
      />
      <SettingToggle
        checked={workflowValidateConnections}
        description="Reject connections between incompatible field types while wiring nodes. Turn off to wire anything (runs may fail)."
        label="Validate connections"
        onChange={updateWorkflowValidateConnections}
      />
    </SettingsSection>
  );
};

const QueueSection = () => {
  const { queueJobsScope } = useWorkbenchPreferences();
  const updateQueueJobsScope = useCallback((value: string) => {
    updatePreferences({ queueJobsScope: value === 'active-project' ? 'active-project' : 'all' });
  }, []);

  return (
    <SettingsSection description="Choose which jobs the Queue widget includes in its counts and lists." title="Queue">
      <SettingSelect
        description="Show jobs from only the active project, or all queue jobs visible to you."
        label="Show jobs from"
        value={queueJobsScope}
        onChange={updateQueueJobsScope}
      >
        <option value="active-project">Active project</option>
        <option value="all">All</option>
      </SettingSelect>
      <ClearIntermediatesSetting />
    </SettingsSection>
  );
};

const DeveloperSection = () => {
  const { developerLogEnabled, developerLogLevel, developerLogNamespaces, developerPerformanceTimingsEnabled } =
    useWorkbenchPreferences();
  const enabledNamespaces = useMemo(() => new Set(developerLogNamespaces), [developerLogNamespaces]);

  const toggleNamespace = useCallback(
    (namespace: DeveloperLogNamespace, checked: boolean) => {
      const next = checked
        ? [...developerLogNamespaces, namespace]
        : developerLogNamespaces.filter((candidate) => candidate !== namespace);

      updatePreferences({
        developerLogNamespaces: DEVELOPER_LOG_NAMESPACES.filter((candidate) => next.includes(candidate)),
      });
    },
    [developerLogNamespaces]
  );
  const updateDeveloperLogEnabled = useCallback((checked: boolean) => {
    updatePreferences({ developerLogEnabled: checked });
  }, []);
  const updateDeveloperLogLevel = useCallback((value: string) => {
    updatePreferences({ developerLogLevel: value as DeveloperLogLevel });
  }, []);
  const updateDeveloperPerformanceTimingsEnabled = useCallback((checked: boolean) => {
    updatePreferences({ developerPerformanceTimingsEnabled: checked });
  }, []);

  return (
    <SettingsSection
      description="Current-user diagnostics settings. Entries are still grouped by project and kept in memory only."
      title="Developer"
    >
      <SettingToggle
        checked={developerLogEnabled}
        description="Record selected diagnostic log events in the Diagnostics widget."
        label="Record diagnostic logs"
        onChange={updateDeveloperLogEnabled}
      />
      <SettingSelect
        description="Minimum log level recorded in the Diagnostics widget."
        label="Log level"
        value={developerLogLevel}
        onChange={updateDeveloperLogLevel}
      >
        {DEVELOPER_LOG_LEVELS.map((value) => (
          <option key={value} value={value}>
            {formatSettingLabel(value)}
          </option>
        ))}
      </SettingSelect>
      <SettingToggle
        checked={developerPerformanceTimingsEnabled}
        description="Record performance measurements such as workflow editor timing and project serialization costs."
        label="Collect performance timings"
        onChange={updateDeveloperPerformanceTimingsEnabled}
      />
      <Stack gap="2">
        <Text color="fg" fontSize="sm" fontWeight="500">
          Log namespaces
        </Text>
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
      </Stack>
    </SettingsSection>
  );
};

const DeveloperNamespaceCheckbox = ({
  checked,
  namespace,
  toggleNamespace,
}: {
  checked: boolean;
  namespace: DeveloperLogNamespace;
  toggleNamespace: (namespace: DeveloperLogNamespace, checked: boolean) => void;
}) => {
  const handleCheckedChange = useCallback(
    (event: { checked: boolean | 'indeterminate' }) => toggleNamespace(namespace, event.checked === true),
    [namespace, toggleNamespace]
  );

  return (
    <Checkbox.Root checked={checked} size="sm" onCheckedChange={handleCheckedChange}>
      <Checkbox.HiddenInput />
      <Checkbox.Control />
      <Checkbox.Label color="fg.muted" fontSize="xs">
        {formatSettingLabel(namespace)}
      </Checkbox.Label>
    </Checkbox.Root>
  );
};

const WorkspaceSection = () => {
  const dispatch = useOptionalWorkbenchDispatch();
  const { scope } = useWorkbenchSettings();
  const [isClearConfirmOpen, setIsClearConfirmOpen] = useState(false);

  const clearSavedData = useCallback(async () => {
    await Promise.all([syncedWorkbenchPersistence.clearWorkbench(), clearWorkbenchSettings()]);
    window.location.reload();
  }, []);
  const resetLayout = useCallback(() => dispatch?.({ type: 'resetActiveLayout' }), [dispatch]);
  const openClearConfirm = useCallback(() => setIsClearConfirmOpen(true), []);
  const closeClearConfirm = useCallback(() => setIsClearConfirmOpen(false), []);

  return (
    <SettingsSection
      description="Reset the layout, or permanently delete saved projects and settings."
      title="Workspace"
    >
      <HStack gap="2" wrap="wrap">
        {dispatch ? (
          <Button size="sm" variant="outline" onClick={resetLayout}>
            <RotateCcwIcon />
            Reset layout
          </Button>
        ) : null}
        <Button
          borderColor="border.emphasized"
          color="fg.error"
          size="sm"
          variant="outline"
          _hover={DANGER_BUTTON_HOVER_STYLES}
          onClick={openClearConfirm}
        >
          <Trash2Icon />
          Clear saved data…
        </Button>
      </HStack>
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
    </SettingsSection>
  );
};

const SettingToggle = ({
  checked,
  comingSoon,
  description,
  label,
  onChange,
}: {
  checked: boolean;
  comingSoon?: boolean;
  description: string;
  label: string;
  onChange: (checked: boolean) => void;
}) => {
  const descriptionId = useId();
  const handleCheckedChange = useCallback(
    (event: { checked: boolean | 'indeterminate' }) => onChange(event.checked === true),
    [onChange]
  );

  return (
    <Switch.Root
      alignItems="center"
      checked={checked}
      disabled={comingSoon}
      display="flex"
      gap="4"
      justifyContent="space-between"
      w="full"
      onCheckedChange={handleCheckedChange}
    >
      <Stack gap="0.5">
        <Switch.Label color="fg" fontSize="sm" fontWeight="500" m="0">
          {label}
        </Switch.Label>
        <Text color="fg.subtle" fontSize="xs" id={descriptionId}>
          {description}
        </Text>
      </Stack>
      <Switch.HiddenInput aria-describedby={descriptionId} />
      <Switch.Control flexShrink={0} _checked={SWITCH_CHECKED_STYLES}>
        <Switch.Thumb />
      </Switch.Control>
    </Switch.Root>
  );
};

const SettingSelect = ({
  children,
  comingSoon,
  description,
  label,
  onChange,
  value,
}: {
  children: ReactNode;
  comingSoon?: boolean;
  description: string;
  label: string;
  onChange: (value: string) => void;
  value: string;
}) => {
  const handleChange = useCallback(
    (event: { currentTarget: { value: string } }) => onChange(event.currentTarget.value),
    [onChange]
  );

  return (
    <Field.Root
      alignItems={FIELD_ALIGN_ITEMS}
      disabled={comingSoon}
      display="flex"
      flexDirection={FIELD_FLEX_DIRECTION}
      gap="3"
      justifyContent="space-between"
    >
      <Stack gap="0.5">
        <Field.Label color="fg" fontSize="sm" fontWeight="500" m="0">
          {label}
        </Field.Label>
        <Field.HelperText color="fg.subtle" fontSize="xs" m="0">
          {description}
        </Field.HelperText>
      </Stack>
      <NativeSelect.Root flexShrink={0} maxW={SELECT_MAX_WIDTH} size="sm" w="full">
        <NativeSelect.Field value={value} onChange={handleChange}>
          {children}
        </NativeSelect.Field>
        <NativeSelect.Indicator />
      </NativeSelect.Root>
    </Field.Root>
  );
};

const formatSettingLabel = (value: string): string =>
  value
    .split('-')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
