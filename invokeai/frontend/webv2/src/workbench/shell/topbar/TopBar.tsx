import { Grid, HStack } from '@chakra-ui/react';
import { PaletteButton } from '@workbench/palette/PaletteButton';
import { SettingsDialogHost } from '@workbench/settings/SettingsDialogHost';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { WidgetsDialog } from '@workbench/shell/WidgetsDialog';

import { AppMenu } from './AppMenu';
import { InvocationCluster } from './InvocationCluster';
import { LayoutPresetAdminDialogs } from './LayoutPresetAdminDialogs';
import { LayoutPresetManagerDialog } from './LayoutPresetManagerDialog';
import { LayoutPresetStrip } from './LayoutPresetStrip';
import { ProjectSwitcher } from './ProjectSwitcher';
import { QueueCluster } from './QueueCluster';
import { TopbarProgressRail } from './TopbarProgressRail';

/**
 * Left-side Invoke controls precede the project switcher; the menu and queue stay at the outer edges.
 */
const TOPBAR_COLUMNS = 'auto auto minmax(0, 1fr) auto minmax(max-content, 1fr) auto';

export const TopBar = () => {
  const showInvokeButtonOnLeft = useWorkbenchPreferenceSelector((preferences) => preferences.showInvokeButtonOnLeft);
  return (
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
        <HStack gridColumn="1" gridRow="1">
          <AppMenu />
        </HStack>
        <HStack gap="0.5" minW="0" maxW="min(22rem, 24vw)" gridColumn="3" gridRow="1" justifySelf="start">
          <ProjectSwitcher />
          <PaletteButton />
        </HStack>

        <HStack gridColumn="4" gridRow="1" minW="0">
          <LayoutPresetStrip />
        </HStack>

        <HStack
          data-invocation-side={showInvokeButtonOnLeft ? 'left' : 'right'}
          gridColumn={showInvokeButtonOnLeft ? '2' : '5'}
          gridRow="1"
          justifySelf={showInvokeButtonOnLeft ? 'start' : 'end'}
          minW="max-content"
        >
          <InvocationCluster />
        </HStack>
        <HStack gridColumn="6" gridRow="1">
          <QueueCluster />
        </HStack>

        <TopbarProgressRail />
      </Grid>
      {/* Dialog hosts, not controls: every surface that can open one writes to a
        store, so the body cannot belong to whichever trigger happens to be
        mounted. */}
      <LayoutPresetManagerDialog />
      <LayoutPresetAdminDialogs />
      <SettingsDialogHost />
      <WidgetsDialog />
    </>
  );
};
