import {
  Box,
  chakra,
  Dialog,
  Flex,
  HStack,
  Icon,
  Portal,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  useSlotRecipe,
} from '@chakra-ui/react';
import type { ReactNode } from 'react';
import { PiArrowCounterClockwiseBold, PiCheckBold, PiGearSixBold, PiTrashBold } from 'react-icons/pi';

import { themeCardRecipe } from '../../theme/recipes';
import { THEMES, type ThemeDefinition } from '../../theme/system';
import { localStorageWorkbenchPersistence } from '../persistence';
import type { WorkbenchThemeId } from '../types';
import { useWorkbench } from '../WorkbenchContext';
import { Button, CloseButton, IconButton } from './ui/Button';

/**
 * Settings entry point: a self-contained gear button that owns its own dialog
 * open-state. Kept as one composable unit (rather than threading an `isOpen`
 * boolean through the TopBar) so the rest of the shell stays unaware of it.
 */
export const SettingsButton = () => (
  <Dialog.Root placement="center" scrollBehavior="inside" size="lg">
    <Dialog.Trigger asChild>
      <IconButton aria-label="Settings" color="fg.muted" size="sm" variant="ghost" _hover={{ color: 'fg.default' }}>
        <PiGearSixBold />
      </IconButton>
    </Dialog.Trigger>
    <SettingsDialogContent />
  </Dialog.Root>
);

const SettingsDialogContent = () => (
  <Portal>
    <Dialog.Backdrop />
    <Dialog.Positioner>
      <Dialog.Content bg="bg.surface" borderColor="border.subtle" borderWidth="1px" color="fg.default">
        <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
          <Stack gap="0.5">
            <Dialog.Title fontSize="md" fontWeight="700">
              Settings
            </Dialog.Title>
            <Text color="fg.subtle" fontSize="xs">
              Appearance and workspace preferences for this browser.
            </Text>
          </Stack>
        </Dialog.Header>
        <Dialog.Body>
          <Stack gap="7" py="2">
            <AppearanceSection />
            <BehaviorSection />
            <WorkspaceSection />
          </Stack>
        </Dialog.Body>
        <Dialog.CloseTrigger asChild>
          <CloseButton color="fg.muted" size="sm" />
        </Dialog.CloseTrigger>
      </Dialog.Content>
    </Dialog.Positioner>
  </Portal>
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
      <Text color="fg.default" fontSize="sm" fontWeight="600">
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
  const { dispatch, state } = useWorkbench();
  const { reduceMotion, showFocusRegionHighlight, themeId } = state.account.preferences;

  const selectTheme = (nextThemeId: WorkbenchThemeId) => {
    dispatch({ preferences: { themeId: nextThemeId }, type: 'setPreferences' });
  };

  return (
    <SettingsSection description="Choose a theme and tune motion." title="Appearance">
      <SimpleGrid columns={{ base: 2, md: 3 }} gap="3">
        {THEMES.map((theme) => (
          <ThemeCard key={theme.id} selected={theme.id === themeId} theme={theme} onSelect={selectTheme} />
        ))}
      </SimpleGrid>
      <SettingToggle
        checked={reduceMotion}
        description="Disable transitions and animations across the workbench."
        label="Reduce motion"
        onChange={(checked) => dispatch({ preferences: { reduceMotion: checked }, type: 'setPreferences' })}
      />
      <SettingToggle
        checked={showFocusRegionHighlight}
        description="Draw a subtle outline around the active workbench region."
        label="Highlight focused region"
        onChange={(checked) => dispatch({ preferences: { showFocusRegionHighlight: checked }, type: 'setPreferences' })}
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

  return (
    <chakra.button type="button" aria-pressed={selected} css={styles.root} onClick={() => onSelect(theme.id)}>
      <Flex css={styles.preview}>
        <Box css={styles.swatch} bg={theme.colors.surface} />
        <Box css={styles.swatch} bg={theme.colors.panel} />
        <Box css={styles.swatch} bg={theme.colors.accent} />
        <Box css={styles.swatch} bg={theme.colors.active} />
      </Flex>
      <Box css={styles.body}>
        <HStack justify="space-between" w="full">
          <Text css={styles.name}>{theme.label}</Text>
          <Box css={styles.indicator}>
            <Icon as={PiCheckBold} boxSize="3" />
          </Box>
        </HStack>
        <Text css={styles.description}>{theme.description}</Text>
      </Box>
    </chakra.button>
  );
};

const SettingToggle = ({
  checked,
  description,
  label,
  onChange,
}: {
  checked: boolean;
  description: string;
  label: string;
  onChange: (checked: boolean) => void;
}) => (
  <Switch.Root
    alignItems="center"
    checked={checked}
    display="flex"
    justifyContent="space-between"
    w="full"
    onCheckedChange={(event) => onChange(event.checked)}
  >
    <Stack gap="0.5">
      <Switch.Label color="fg.default" fontSize="sm" fontWeight="500" m="0">
        {label}
      </Switch.Label>
      <Text color="fg.subtle" fontSize="xs">
        {description}
      </Text>
    </Stack>
    <Switch.HiddenInput />
    <Switch.Control _checked={{ bg: 'accent.invoke' }}>
      <Switch.Thumb />
    </Switch.Control>
  </Switch.Root>
);

const BehaviorSection = () => {
  const { dispatch, state } = useWorkbench();
  const { confirmImageDeletion } = state.account.preferences;

  return (
    <SettingsSection description="Safety checks and interaction behavior." title="Behavior">
      <SettingToggle
        checked={confirmImageDeletion}
        description="Ask for confirmation before permanently deleting images."
        label="Confirm image deletion"
        onChange={(checked) => dispatch({ preferences: { confirmImageDeletion: checked }, type: 'setPreferences' })}
      />
    </SettingsSection>
  );
};

const WorkspaceSection = () => {
  const { dispatch } = useWorkbench();

  const clearSavedData = () => {
    void localStorageWorkbenchPersistence.clearWorkbench().then(() => {
      window.location.reload();
    });
  };

  return (
    <SettingsSection description="Reset the layout or clear locally saved projects and preferences." title="Workspace">
      <HStack gap="2" wrap="wrap">
        <Button size="sm" variant="outline" onClick={() => dispatch({ type: 'resetActiveLayout' })}>
          <PiArrowCounterClockwiseBold />
          Reset layout
        </Button>
        <Button
          borderColor="border.emphasis"
          color="fg.error"
          size="sm"
          variant="outline"
          _hover={{ bg: 'fg.error', color: 'bg.surface' }}
          onClick={clearSavedData}
        >
          <PiTrashBold />
          Clear saved data
        </Button>
      </HStack>
    </SettingsSection>
  );
};
