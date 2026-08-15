import { Grid, HStack } from '@chakra-ui/react';
import { SettingsDialogHost } from '@workbench/settings/SettingsDialogHost';

import { AppMenu } from './AppMenu';
import { InvocationCluster } from './InvocationCluster';
import { LayoutPresetAdminDialogs } from './LayoutPresetAdminDialogs';
import { LayoutPresetManagerDialog } from './LayoutPresetManagerDialog';
import { LayoutPresetStrip } from './LayoutPresetStrip';
import { ProjectSwitcher } from './ProjectSwitcher';

/**
 * The workbench top bar.
 *
 * Three columns, not a flex row: the `1fr` sides let the preset strip stay
 * geometrically centred no matter how long the project name is. A flex layout moves
 * the strip horizontally every time the user switches projects, which is highly
 * visible on a bar that is always on screen. `minmax(0, 1fr)` rather than `1fr`
 * so the left column can actually shrink and truncate instead of pushing the
 * grid wider than its container.
 *
 * The zones degrade from the left: labels, then the project name, before
 * anything in the invocation cluster gives way. The routing indicator and the
 * queue readout never collapse at any width (§10).
 */
const TOPBAR_COLUMNS = 'minmax(0, 1fr) auto minmax(0, 1fr)';

export const TopBar = () => (
  <>
    <Grid
      alignItems="center"
      as="header"
      bg="bg.subtle"
      borderBottomWidth="1px"
      borderColor="border.subtle"
      flexShrink={0}
      gap="2"
      gridTemplateColumns={TOPBAR_COLUMNS}
      h="44px"
      px="1.5"
      w="full"
    >
      <HStack gap="0.5" minW="0">
        <AppMenu />
        <ProjectSwitcher />
      </HStack>

      <LayoutPresetStrip />

      <InvocationCluster />
    </Grid>
    {/* Dialog hosts, not controls: every surface that can open one writes to a
        store, so the body cannot belong to whichever trigger happens to be
        mounted. */}
    <LayoutPresetManagerDialog />
    <LayoutPresetAdminDialogs />
    <SettingsDialogHost />
  </>
);
