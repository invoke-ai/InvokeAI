/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Box, Flex, Icon, Text } from '@chakra-ui/react';
import { useCustomNodesSelector } from '@features/nodes/data/nodesStore';
import { NodeActivityBar } from '@features/nodes/ui/activity/NodeActivityBar';
import { AddNodesView } from '@features/nodes/ui/add-nodes/AddNodesView';
import { NodePackDetail } from '@features/nodes/ui/detail/NodePackDetail';
import { updateNodesUi, useNodesUiSelector, type NodesManagerTab } from '@features/nodes/ui/nodesUiStore';
import { Scrollable, Tabs } from '@platform/ui';
import { BlocksIcon, PlusIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { HEADER_MIN_HEIGHT } from './layoutConstants';

/** Right side of the nodes manager: selected pack details, Add Nodes, and activity footer. */
export const DetailPane = () => {
  const { t } = useTranslation();
  const activePackName = useNodesUiSelector((snapshot) => snapshot.activePackName);
  const activeTab = useNodesUiSelector((snapshot) => snapshot.activeTab);
  const nodePacks = useCustomNodesSelector((snapshot) => snapshot.nodePacks);
  const activePack = nodePacks.find((pack) => pack.name === activePackName) ?? null;
  const detailLabel = activePack?.name ?? t('nodes.details');

  return (
    <Tabs.Root
      asChild
      lazyMount
      size="sm"
      unmountOnExit
      value={activeTab}
      onValueChange={(event) => updateNodesUi({ activeTab: event.value as NodesManagerTab })}
    >
      <Flex direction="column" flex="1" minH="0" minW="0">
        <Flex align="flex-end" borderBottomWidth={1} flexShrink={0} minH={HEADER_MIN_HEIGHT} px="2">
          <Tabs.List mb="-1px">
            <Tabs.Trigger fontSize="xs" value="details">
              <Icon as={BlocksIcon} boxSize="3" />
              <Text maxW="14rem" truncate>
                {detailLabel}
              </Text>
            </Tabs.Trigger>
            <Tabs.Trigger fontSize="xs" value="add">
              <Icon as={PlusIcon} boxSize="3" />
              {t('nodes.addNodes')}
            </Tabs.Trigger>
          </Tabs.List>
        </Flex>

        <Box flex="1" minH="0">
          <Tabs.Content h="full" p="0" value="details">
            <DetailTab packName={activePackName} />
          </Tabs.Content>
          <Tabs.Content h="full" p="0" value="add">
            <AddNodesView />
          </Tabs.Content>
        </Box>

        <NodeActivityBar />
      </Flex>
    </Tabs.Root>
  );
};

const DetailTab = ({ packName }: { packName: string | null }) => {
  const { t } = useTranslation();
  const nodePacks = useCustomNodesSelector((snapshot) => snapshot.nodePacks);
  const activePack = nodePacks.find((pack) => pack.name === packName) ?? null;

  if (!activePack) {
    return (
      <Flex align="center" direction="column" gap="2" h="full" justify="center" p="6">
        <Icon as={BlocksIcon} boxSize="8" color="fg.subtle" />
        <Text color="fg.muted" fontSize="sm" fontWeight="600">
          {t('nodes.selectPack')}
        </Text>
        <Text color="fg.muted" fontSize="xs" maxW="22rem" textAlign="center">
          {t('nodes.selectPackDescription')}
        </Text>
      </Flex>
    );
  }

  return (
    <Scrollable h="full" label={t('nodes.details')} minH="0" p="3">
      <NodePackDetail pack={activePack} onUninstalled={() => updateNodesUi({ activePackName: null })} />
    </Scrollable>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
