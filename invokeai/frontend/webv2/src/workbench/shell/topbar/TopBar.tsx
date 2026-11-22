import { Box, Grid, HStack } from '@chakra-ui/react';
import { PaletteButton } from '@workbench/palette/PaletteButton';
import { SettingsDialogHost } from '@workbench/settings/SettingsDialogHost';
import { WidgetsDialog } from '@workbench/shell/WidgetsDialog';
import { useCustomizeSession } from '@workbench/workspace/useStructureLock';
import { lazy, Suspense } from 'react';

import { AppMenu } from './AppMenu';
import { InvocationCluster } from './InvocationCluster';
import { LayoutPresetAdminDialogs } from './LayoutPresetAdminDialogs';
import { LayoutPresetManagerDialog } from './LayoutPresetManagerDialog';
import { LayoutPresetStrip } from './LayoutPresetStrip';
import { ProjectSwitcher } from './ProjectSwitcher';
import { TopbarProgressRail } from './TopbarProgressRail';

/** Use minmax(0,1fr) side columns to center presets independently of project-name width. */
const TOPBAR_COLUMNS = 'minmax(0, 1fr) auto minmax(0, 1fr)';

/** Loads with the first Customize, outside the editor's initial payload. */
const LazyCustomizeBar = lazy(() => import('./CustomizeBar').then((module) => ({ default: module.CustomizeBar })));

// Holds the bar's height while its module loads, so the row does not jump.
const CUSTOMIZE_BAR_PENDING = <Box h="9" />;

/** The workspace tabs, or the Customize bar in their place while the active project customizes its workspace. */
const WorkspaceStrip = () => {
  const session = useCustomizeSession();

  return session ? (
    <Suspense fallback={CUSTOMIZE_BAR_PENDING}>
      <LazyCustomizeBar session={session} />
    </Suspense>
  ) : (
    <LayoutPresetStrip />
  );
};

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
      // Positioning context for the progress rail, which overlays the bottom border rather than taking a row of its
      // own.
      position="relative"
      px="1.5"
      w="full"
    >
      <HStack gap="0.5" minW="0">
        <AppMenu />
        <ProjectSwitcher />
        {/* The palette's own button, beside the project, so its shortcut is not the only way in. */}
        <PaletteButton />
      </HStack>

      <WorkspaceStrip />

      <InvocationCluster />

      <TopbarProgressRail />
    </Grid>
    {/* Dialog hosts, not controls: every surface that can open one writes to a store, so the body cannot belong to
        whichever trigger happens to be mounted. */}
    <LayoutPresetManagerDialog />
    <LayoutPresetAdminDialogs />
    <SettingsDialogHost />
    <WidgetsDialog />
  </>
);
