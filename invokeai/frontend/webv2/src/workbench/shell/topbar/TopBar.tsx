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
 * Equal side columns keep presets centered independently of Invoke placement and project-name width.
 */
const TOPBAR_COLUMNS = 'minmax(0, 1fr) auto minmax(0, 1fr)';

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
        <HStack gap="0.5" minW="0">
          <AppMenu />
          {showInvokeButtonOnLeft ? (
            <HStack data-invocation-side="left" flexShrink={0}>
              <InvocationCluster />
            </HStack>
          ) : null}
          <ProjectSwitcher />
          <PaletteButton />
        </HStack>

        <LayoutPresetStrip />

        <HStack gap="2" justify="end" minW="0">
          {showInvokeButtonOnLeft ? null : (
            <HStack data-invocation-side="right" flexShrink={0}>
              <InvocationCluster />
            </HStack>
          )}
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
