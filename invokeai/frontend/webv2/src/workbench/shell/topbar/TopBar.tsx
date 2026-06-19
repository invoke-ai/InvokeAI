import { Box, Flex, HStack, Separator } from '@chakra-ui/react';
import { useCapabilities } from '@workbench/auth/capabilities';
import { AccountMenu } from '@workbench/auth/components/AccountMenu';
import { ProjectTabs } from '@workbench/projects/components';

import { SettingsButton } from '@/workbench/settings';

import { BatchCountField } from './BatchCountField';
import { BrandMark } from './BrandMark';
import { InvokeControl } from './InvokeControl';
import { LayoutPresetMenu } from './LayoutPresetMenu';
import { ModelManagerButton } from './ModelManagerButton';
import { NodesManagerButton } from './NodeManagerButton';
import { QueueActions } from './QueueActions';
import { QueueInfo } from './QueueInfo';

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
      <Box h="12" w="12">
        <BrandMark />
      </Box>
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
        <SettingsButton />
      </HStack>
      <AccountMenu />
    </Flex>
  );
};
