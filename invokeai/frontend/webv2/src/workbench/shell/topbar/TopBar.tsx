import { Grid, HStack } from '@chakra-ui/react';
import { SettingsDialogHost } from '@workbench/settings/SettingsDialogHost';

import { AppMenu } from './AppMenu';
import { InvocationCluster } from './InvocationCluster';
import { LayoutPresetAdminDialogs } from './LayoutPresetAdminDialogs';
import { LayoutPresetManagerDialog } from './LayoutPresetManagerDialog';
import { LayoutPresetStrip } from './LayoutPresetStrip';
import { ProjectSwitcher } from './ProjectSwitcher';
import { TopbarProgressRail } from './TopbarProgressRail';

/**
 * Use minmax(0,1fr) side columns to center presets independently of project-name width. Collapse left labels
 * first; preserve route and queue indicators.
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
      // Positioning context for the progress rail, which overlays the bottom
      // border rather than taking a row of its own.
      position="relative"
      px="1.5"
      w="full"
    >
      <HStack gap="0.5" minW="0">
        <AppMenu />
        <ProjectSwitcher />
      </HStack>

      <LayoutPresetStrip />

      <InvocationCluster />

      <TopbarProgressRail />
    </Grid>
    {/* Dialog hosts, not controls: every surface that can open one writes to a
        store, so the body cannot belong to whichever trigger happens to be
        mounted. */}
    <LayoutPresetManagerDialog />
    <LayoutPresetAdminDialogs />
    <SettingsDialogHost />
  </>
);
