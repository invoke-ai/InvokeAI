import {
  Box,
  chakra,
  Checkbox,
  Dialog,
  Field,
  Flex,
  HStack,
  Icon,
  NativeSelect,
  Portal,
  SimpleGrid,
  Stack,
  Switch,
  Tabs,
  Text,
  useSlotRecipe,
} from '@chakra-ui/react';
import { useId, useState, type ReactNode } from 'react';
import {
  CheckIcon,
  Code2Icon,
  DatabaseIcon,
  FolderIcon,
  PaletteIcon,
  RotateCcwIcon,
  SettingsIcon,
  SlidersHorizontalIcon,
  Trash2Icon,
  WorkflowIcon,
  type LucideIcon,
} from 'lucide-react';

import { themeCardRecipe } from '@theme/recipes';
import { previewSwatches, THEMES, type ThemeDefinition } from '@theme/system';
import { Button, CloseButton, IconButton } from '@workbench/components/ui/Button';
import { ConfirmDialog } from '@workbench/components/ui/ConfirmDialog';
import { syncedWorkbenchPersistence } from '@workbench/projects/syncedPersistence';
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
  useOptionalWorkbenchDispatch,
  useOptionalWorkbenchSelector,
  useOptionalWorkbenchStore,
} from '@workbench/WorkbenchContext';
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

/**
 * The top bar's settings entry point. It also hosts the dialog itself, driven
 * by `settingsDialogStore` so any surface can deep-link into a section via
 * `openWorkbenchSettings('workflow')`.
 */
export const SettingsButton = () => {
  const { isOpen } = settingsDialogStore.useSnapshot();

  return (
    <>
      <IconButton
        aria-label="Settings"
        color="fg.muted"
        size="sm"
        variant="ghost"
        _hover={{ color: 'fg' }}
        onClick={() => openWorkbenchSettings()}
      >
        <SettingsIcon />
      </IconButton>
      <SettingsDialog isOpen={isOpen} onClose={closeWorkbenchSettings} />
    </>
  );
};

export const SettingsDialog = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => (
  <Dialog.Root
    lazyMount
    open={isOpen}
    placement="center"
    scrollBehavior="inside"
    size="xl"
    unmountOnExit
    onOpenChange={(event) => {
      if (!event.open) {
        onClose();
      }
    }}
  >
    <SettingsDialogContent />
  </Dialog.Root>
);

const SettingsDialogContent = () => {
  const { error, scope, status } = useWorkbenchSettings();

  return (
    <Portal>
      <Dialog.Backdrop />
      <Dialog.Positioner>
        <Dialog.Content h="min(46rem, calc(100dvh - 4rem))" maxW="4xl">
          <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
            <Stack gap="0.5">
              <Dialog.Title fontSize="md" fontWeight="700">
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
          </Dialog.Header>
          <Dialog.Body minH="0" p="0">
            <SettingsTabs />
          </Dialog.Body>
          <Dialog.CloseTrigger asChild>
            <CloseButton color="fg.muted" size="sm" />
          </Dialog.CloseTrigger>
        </Dialog.Content>
      </Dialog.Positioner>
    </Portal>
  );
};

const SettingsTabs = () => {
  const hasWorkbench = useOptionalWorkbenchStore() !== null;
  const allTabs: SettingsTabDefinition[] = [
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
      children: <ProjectSection />,
      condition: hasWorkbench,
      icon: FolderIcon,
      label: 'Project',
      value: 'project',
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
  ];
  const tabs = allTabs.filter((tab) => tab.condition !== false);
  const { sectionId } = settingsDialogStore.useSnapshot();
  const activeSectionId = tabs.some((tab) => tab.value === sectionId) ? sectionId : 'appearance';

  return (
    <Tabs.Root
      display="flex"
      h="full"
      minH="0"
      orientation="vertical"
      value={activeSectionId}
      variant="subtle"
      onValueChange={(event) => setWorkbenchSettingsSection(event.value as SettingsSectionId)}
    >
      <Tabs.List
        alignItems="stretch"
        bg="bg"
        borderColor="border.subtle"
        borderRightWidth="1px"
        flexShrink={0}
        gap="1"
        p="2"
        w={{ base: '40', md: '52' }}
      >
        {tabs.map((tab) => (
          <SettingsTabTrigger key={tab.value} icon={tab.icon} label={tab.label} value={tab.value} />
        ))}
      </Tabs.List>
      <Box flex="1" minW="0" overflowY="auto" p={{ base: '4', md: '5' }}>
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
  <Tabs.Trigger
    color="fg.muted"
    fontSize="xs"
    fontWeight="600"
    gap="2"
    justifyContent="flex-start"
    px="3"
    py="2"
    rounded="md"
    textAlign="start"
    value={value}
    w="full"
    _selected={{ bg: 'bg.muted', color: 'fg' }}
  >
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

  const selectTheme = (nextThemeId: WorkbenchThemeId) => {
    updatePreferences({ themeId: nextThemeId });
  };

  return (
    <SettingsSection description="Choose a theme, language, and motion behavior." title="Appearance">
      <SimpleGrid columns={{ base: 2, md: 3 }} gap="3">
        {THEMES.map((theme) => (
          <ThemeCard key={theme.id} selected={theme.id === themeId} theme={theme} onSelect={selectTheme} />
        ))}
      </SimpleGrid>
      <SettingSelect
        comingSoon
        description="Interface language. More i18n coverage will arrive soon."
        label="Language"
        value={language}
        onChange={(value) => updatePreferences({ language: value as WorkbenchLanguage })}
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
        onChange={(checked) => updatePreferences({ reduceMotion: checked })}
      />
      <SettingToggle
        checked={showFocusRegionHighlight}
        description="Draw a subtle outline around the active workbench region."
        label="Highlight focused region"
        onChange={(checked) => updatePreferences({ showFocusRegionHighlight: checked })}
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

  return (
    <chakra.button type="button" aria-pressed={selected} css={styles.root} onClick={() => onSelect(theme.id)}>
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

  return (
    <SettingsSection description="Safety checks and user-assistance behavior." title="Behavior">
      <SettingToggle
        checked={confirmImageDeletion}
        description="Ask for confirmation before permanently deleting images."
        label="Confirm image deletion"
        onChange={(checked) => updatePreferences({ confirmImageDeletion: checked })}
      />
      <SettingToggle
        checked={enableInformationalPopovers}
        comingSoon
        description="Show educational popovers on controls that have extra guidance."
        label="Enable informational popovers"
        onChange={(checked) => updatePreferences({ enableInformationalPopovers: checked })}
      />
      <SettingToggle
        checked={enableModelDescriptions}
        description="Include model descriptions in model dropdowns where available."
        label="Enable model descriptions in dropdowns"
        onChange={(checked) => updatePreferences({ enableModelDescriptions: checked })}
      />
    </SettingsSection>
  );
};

const ProjectSection = () => {
  const activeProject = useOptionalWorkbenchSelector<Project | null>((snapshot) => snapshot.activeProject, null);
  const dispatch = useOptionalWorkbenchDispatch();

  if (!activeProject || !dispatch) {
    return null;
  }

  const settings = activeProject.settings;
  const updateProjectSettings = (patch: Partial<ProjectSettings>) => {
    dispatch({ settings: patch, type: 'setActiveProjectSettings' });
  };

  return (
    <SettingsSection
      description={`Settings saved with ${activeProject.name}. Future project-only settings can live here without becoming user preferences.`}
      title="Project"
    >
      <SettingToggle
        checked={settings.useCpuNoise}
        description="Use CPU noise generation for deterministic legacy-compatible outputs."
        label="Use CPU noise"
        onChange={(checked) => updateProjectSettings({ useCpuNoise: checked })}
      />
      <SettingToggle
        checked={settings.showProgressDetails}
        comingSoon
        description="Show detailed invocation progress when the backend reports it."
        label="Show progress details"
        onChange={(checked) => updateProjectSettings({ showProgressDetails: checked })}
      />
      <SettingToggle
        checked={settings.antialiasProgressImages}
        comingSoon
        description="Smooth progress previews instead of rendering them pixelated."
        label="Antialias progress images"
        onChange={(checked) => updateProjectSettings({ antialiasProgressImages: checked })}
      />
      <SettingToggle
        checked={settings.showProgressImagesInViewer}
        comingSoon
        description="Show progress images in the viewer when an image is still generating."
        label="Show progress images in viewer"
        onChange={(checked) => updateProjectSettings({ showProgressImagesInViewer: checked })}
      />
      <SettingToggle
        checked={settings.preferNumericAttentionStyle}
        comingSoon
        description="Prefer numeric prompt attention syntax when controls insert attention weights."
        label="Prefer numeric attention style"
        onChange={(checked) => updateProjectSettings({ preferNumericAttentionStyle: checked })}
      />
    </SettingsSection>
  );
};

const WorkflowSection = () => {
  const { workflowEdgeStyle, workflowShowMinimap, workflowSnapToGrid, workflowValidateConnections } =
    useWorkbenchPreferences();

  return (
    <SettingsSection description="Editing behavior for the project graph workflow editor." title="Workflow">
      <SettingSelect
        description="How connections between nodes are drawn in the editor."
        label="Connection style"
        value={workflowEdgeStyle}
        onChange={(value) => updatePreferences({ workflowEdgeStyle: value === 'straight' ? 'straight' : 'curved' })}
      >
        <option value="curved">Curved</option>
        <option value="straight">Straight</option>
      </SettingSelect>
      <SettingToggle
        checked={workflowSnapToGrid}
        description="Snap nodes to the grid while dragging. When off, hold Ctrl to snap temporarily."
        label="Always snap to grid"
        onChange={(checked) => updatePreferences({ workflowSnapToGrid: checked })}
      />
      <SettingToggle
        checked={workflowShowMinimap}
        description="Show the minimap overview in the corner of the workflow editor."
        label="Show minimap"
        onChange={(checked) => updatePreferences({ workflowShowMinimap: checked })}
      />
      <SettingToggle
        checked={workflowValidateConnections}
        description="Reject connections between incompatible field types while wiring nodes. Turn off to wire anything (runs may fail)."
        label="Validate connections"
        onChange={(checked) => updatePreferences({ workflowValidateConnections: checked })}
      />
    </SettingsSection>
  );
};

const DeveloperSection = () => {
  const { developerLogEnabled, developerLogLevel, developerLogNamespaces } = useWorkbenchPreferences();
  const enabledNamespaces = new Set(developerLogNamespaces);

  const toggleNamespace = (namespace: DeveloperLogNamespace, checked: boolean) => {
    const next = checked
      ? [...developerLogNamespaces, namespace]
      : developerLogNamespaces.filter((candidate) => candidate !== namespace);

    updatePreferences({
      developerLogNamespaces: DEVELOPER_LOG_NAMESPACES.filter((candidate) => next.includes(candidate)),
    });
  };

  return (
    <SettingsSection description="Console logging controls for development builds." title="Developer">
      <SettingToggle
        checked={developerLogEnabled}
        comingSoon
        description="Enable console logging for selected namespaces."
        label="Enable developer logs"
        onChange={(checked) => updatePreferences({ developerLogEnabled: checked })}
      />
      <SettingSelect
        comingSoon
        description="Minimum log level written to the console."
        label="Log level"
        value={developerLogLevel}
        onChange={(value) => updatePreferences({ developerLogLevel: value as DeveloperLogLevel })}
      >
        {DEVELOPER_LOG_LEVELS.map((value) => (
          <option key={value} value={value}>
            {formatSettingLabel(value)}
          </option>
        ))}
      </SettingSelect>
      <Stack gap="2">
        <Text color="fg" fontSize="sm" fontWeight="500">
          Log namespaces
        </Text>
        <SimpleGrid columns={{ base: 1, md: 2 }} gap="2">
          {DEVELOPER_LOG_NAMESPACES.map((namespace) => (
            <Checkbox.Root
              key={namespace}
              checked={enabledNamespaces.has(namespace)}
              disabled
              size="sm"
              onCheckedChange={(event) => toggleNamespace(namespace, event.checked === true)}
            >
              <Checkbox.HiddenInput />
              <Checkbox.Control />
              <Checkbox.Label color="fg.muted" fontSize="xs">
                {formatSettingLabel(namespace)}
              </Checkbox.Label>
            </Checkbox.Root>
          ))}
        </SimpleGrid>
      </Stack>
    </SettingsSection>
  );
};

const WorkspaceSection = () => {
  const dispatch = useOptionalWorkbenchDispatch();
  const { scope } = useWorkbenchSettings();
  const [isClearConfirmOpen, setIsClearConfirmOpen] = useState(false);

  const clearSavedData = async () => {
    await Promise.all([syncedWorkbenchPersistence.clearWorkbench(), clearWorkbenchSettings()]);
    window.location.reload();
  };

  return (
    <SettingsSection
      description="Reset the layout, or permanently delete saved projects and settings."
      title="Workspace"
    >
      <HStack gap="2" wrap="wrap">
        {dispatch ? (
          <Button size="sm" variant="outline" onClick={() => dispatch({ type: 'resetActiveLayout' })}>
            <RotateCcwIcon />
            Reset layout
          </Button>
        ) : null}
        <Button
          borderColor="border.emphasized"
          color="fg.error"
          size="sm"
          variant="outline"
          _hover={{ bg: 'fg.error', color: 'bg.subtle' }}
          onClick={() => setIsClearConfirmOpen(true)}
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
        onClose={() => setIsClearConfirmOpen(false)}
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

  return (
    <Switch.Root
      alignItems="center"
      checked={checked}
      disabled={comingSoon}
      display="flex"
      gap="4"
      justifyContent="space-between"
      w="full"
      onCheckedChange={(event) => onChange(event.checked)}
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
      <Switch.Control flexShrink={0} _checked={{ bg: 'accent.solid' }}>
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
}) => (
  <Field.Root
    alignItems={{ base: 'stretch', md: 'center' }}
    disabled={comingSoon}
    display="flex"
    flexDirection={{ base: 'column', md: 'row' }}
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
    <NativeSelect.Root flexShrink={0} maxW={{ base: 'full', md: '56' }} size="sm" w="full">
      <NativeSelect.Field value={value} onChange={(event) => onChange(event.currentTarget.value)}>
        {children}
      </NativeSelect.Field>
      <NativeSelect.Indicator />
    </NativeSelect.Root>
  </Field.Root>
);

const formatSettingLabel = (value: string): string =>
  value
    .split('-')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
