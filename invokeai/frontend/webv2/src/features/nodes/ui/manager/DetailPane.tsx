/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { NodePackInfo } from '@features/nodes/core/catalog';

import { Box, Flex, Icon, Text } from '@chakra-ui/react';
import { useCustomNodesSelector } from '@features/nodes/data/nodesStore';
import { AddNodesView } from '@features/nodes/ui/add-nodes/AddNodesView';
import { NodePackDetail } from '@features/nodes/ui/detail/NodePackDetail';
import {
  openNodesManagerTab,
  updateNodesUi,
  useNodesUiSelector,
  type NodesManagerTab,
} from '@features/nodes/ui/nodesUiStore';
import { UninstallPackDialog } from '@features/nodes/ui/shared/UninstallPackDialog';
import { Scrollable, Tabs } from '@platform/ui';
import { ManagerDetailHeader } from '@platform/ui/ManagerLayout';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { BlocksIcon, PlusIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

/** Right side of the nodes manager: selected pack details and Add Nodes. Recent activity is the layout's footer. */
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
      size="xl"
      unmountOnExit
      value={activeTab}
      onValueChange={(event) => openNodesManagerTab(event.value as NodesManagerTab)}
    >
      <Flex direction="column" flex="1" minH="0" minW="0">
        <ManagerDetailHeader>
          <Tabs.List mb="-1px">
            <Tabs.Trigger data-manager-item-tab="" value="details">
              <Icon as={BlocksIcon} boxSize="3" />
              <MiddleTruncate maxW="14rem" minW="0" text={detailLabel} />
            </Tabs.Trigger>
            <Tabs.Trigger value="add">
              <Icon as={PlusIcon} boxSize="3" />
              {t('nodes.addNodes')}
            </Tabs.Trigger>
          </Tabs.List>
        </ManagerDetailHeader>

        <Box flex="1" minH="0">
          <Tabs.Content h="full" p="0" value="details">
            <DetailTab activePack={activePack} />
          </Tabs.Content>
          <Tabs.Content h="full" p="0" value="add">
            <AddNodesView />
          </Tabs.Content>
        </Box>
      </Flex>
    </Tabs.Root>
  );
};

const DetailTab = ({ activePack }: { activePack: NodePackInfo | null }) => {
  const { t } = useTranslation();
  // Owned above the detail: an uninstall unmounts it while this dialog is still animating out.
  const [pendingUninstall, setPendingUninstall] = useState<NodePackInfo | null>(null);

  return (
    <>
      {activePack ? (
        <Scrollable h="full" label={t('nodes.details')} minH="0" p="3">
          <NodePackDetail pack={activePack} onRequestUninstall={setPendingUninstall} />
        </Scrollable>
      ) : (
        <Flex align="center" direction="column" gap="2" h="full" justify="center" p="6">
          <Icon as={BlocksIcon} boxSize="8" color="fg.subtle" />
          <Text color="fg.muted" fontSize="lg" fontWeight="600">
            {t('nodes.selectPack')}
          </Text>
          <Text color="fg.muted" fontSize="md" maxW="22rem" textAlign="center">
            {t('nodes.selectPackDescription')}
          </Text>
        </Flex>
      )}
      <UninstallPackDialog
        pack={pendingUninstall}
        onClose={() => setPendingUninstall(null)}
        onUninstalled={() => updateNodesUi({ activePackName: null })}
      />
    </>
  );
};
