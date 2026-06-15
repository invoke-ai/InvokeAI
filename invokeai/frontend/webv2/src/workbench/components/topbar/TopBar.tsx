import { Flex, Separator } from '@chakra-ui/react';
import { AccountMenu } from '@workbench/auth/components/AccountMenu';
import { InvokeControl } from '@workbench/components/InvokeControl';
import { LayoutPresetMenu } from '@workbench/components/LayoutPresetMenu';
import { ProjectTabs } from '@workbench/components/ProjectTabs';

import { BatchCountField } from './BatchCountField';
import { BrandMark } from './BrandMark';
import { QueueActions } from './QueueActions';
import { QueueInfo } from './QueueInfo';

/** Workbench top bar: brand, global Invoke command cluster, project tabs, layout + account controls. */
export const TopBar = () => (
  <Flex
    align="center"
    as="header"
    bg="bg.subtle"
    borderBottomWidth="1px"
    borderColor="border.subtle"
    flexShrink={0}
    gap="2"
    h="12"
    pe="1.5"
    w="full"
  >
    <BrandMark />
    <InvokeControl />
    <BatchCountField />
    <QueueInfo />
    <QueueActions />
    <Separator orientation="vertical" h={5} />
    <ProjectTabs />
    <LayoutPresetMenu />
    <AccountMenu />
  </Flex>
);
