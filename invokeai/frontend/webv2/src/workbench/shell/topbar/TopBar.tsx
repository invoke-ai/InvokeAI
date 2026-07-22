import { Flex, HStack, Separator } from '@chakra-ui/react';
import { AccountMenu, useCapabilities } from '@features/identity';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { Link } from '@tanstack/react-router';
import { PaletteButton } from '@workbench/palette/PaletteButton';
import { ProjectTabs } from '@workbench/projects/components';

import { SettingsButton } from '@/workbench/settings';

import { BatchCountField } from './BatchCountField';
import { InvokeControl } from './InvokeControl';
import { LayoutPresetMenu } from './LayoutPresetMenu';
import { ModelManagerButton } from './ModelManagerButton';
import { NodesManagerButton } from './NodeManagerButton';
import { QueueActions } from './QueueActions';
import { QueueInfo } from './QueueInfo';

const HOME_LINK_STYLE = {
  alignItems: 'center',
  aspectRatio: '1/1',
  display: 'flex',
  flexShrink: 0,
  height: '100%',
  justifyContent: 'center',
};

/** Workbench top bar: brand, global Invoke command cluster, project tabs, layout + account controls. */
export const TopBar = () => {
  const { canManageModels, canManageNodes } = useCapabilities();

  return (
    <Flex
      align="center"
      as="header"
      bg="bg.subtle"
      borderBottomWidth="1px"
      borderColor="border.subtle"
      flexShrink={0}
      gap="2"
      h="12"
      ps="0.5"
      pe="1.5"
      w="full"
    >
      <Link to="/" style={HOME_LINK_STYLE}>
        <InvokeMark size={20} />
      </Link>
      <Separator orientation="vertical" h={5} ms="-2" />
      <InvokeControl />
      <BatchCountField />
      <QueueInfo />
      <QueueActions />
      <Separator orientation="vertical" h={5} />
      <ProjectTabs />
      <LayoutPresetMenu />
      <HStack gap="0.5">
        {canManageNodes ? <NodesManagerButton /> : null}
        {canManageModels ? <ModelManagerButton /> : null}
        <PaletteButton />
        <SettingsButton />
      </HStack>
      <AccountMenu />
    </Flex>
  );
};
