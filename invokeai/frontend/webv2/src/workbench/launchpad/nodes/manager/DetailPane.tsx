/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Box, Flex, Icon, Text } from '@chakra-ui/react';
import { Scrollable, Tabs } from '@workbench/components/ui';
import { useCustomNodesSelector } from '@workbench/customNodes/nodesStore';
import { updateNodesUi, useNodesUiSelector, type NodesManagerTab } from '@workbench/customNodes/nodesUiStore';
import { NodeActivityBar } from '@workbench/launchpad/nodes/activity/NodeActivityBar';
import { AddNodesView } from '@workbench/launchpad/nodes/add-nodes/AddNodesView';
import { NodePackDetail } from '@workbench/launchpad/nodes/detail/NodePackDetail';
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
    <Flex direction="column" flex="1" minH="0" minW="0">
      <Flex align="flex-end" borderBottomWidth={1} flexShrink={0} minH={HEADER_MIN_HEIGHT} px="2">
        <Tabs.Root
          size="sm"
          mb="-1px"
          value={activeTab}
          onValueChange={(event) => updateNodesUi({ activeTab: event.value as NodesManagerTab })}
        >
          <Tabs.List h="full">
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
        </Tabs.Root>
      </Flex>

      <Box flex="1" minH="0">
        {activeTab === 'details' ? <DetailTab packName={activePackName} /> : null}
        {activeTab === 'add' ? <AddNodesView /> : null}
      </Box>

      <NodeActivityBar />
    </Flex>
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
        <Text color="fg.subtle" fontSize="xs" maxW="22rem" textAlign="center">
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
