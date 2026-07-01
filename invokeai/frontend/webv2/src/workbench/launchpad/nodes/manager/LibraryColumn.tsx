/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Box, Flex, HStack, Text } from '@chakra-ui/react';
import { useCustomNodesSelector } from '@workbench/customNodes/nodesStore';
import { openNodePackDetail, updateNodesUi, useNodesUiSelector } from '@workbench/customNodes/nodesUiStore';
import { NodePackList } from '@workbench/launchpad/nodes/library/NodePackList';
import { useTranslation } from 'react-i18next';

import { HEADER_MIN_HEIGHT, PACK_LIBRARY_WIDTH } from './layoutConstants';
import { ReloadNodesButton } from './ReloadNodesButton';

/** Persistent custom-node pack list, matching the model manager's library column. */
export const LibraryColumn = () => {
  const { t } = useTranslation();
  const activePackName = useNodesUiSelector((snapshot) => snapshot.activePackName);
  const searchTerm = useNodesUiSelector((snapshot) => snapshot.searchTerm);
  const error = useCustomNodesSelector((snapshot) => snapshot.error);
  const nodePacks = useCustomNodesSelector((snapshot) => snapshot.nodePacks);
  const status = useCustomNodesSelector((snapshot) => snapshot.status);

  return (
    <Flex
      borderEndWidth={1}
      direction="column"
      flexShrink={0}
      h="full"
      minH="0"
      position="relative"
      w={PACK_LIBRARY_WIDTH}
    >
      <HStack align="center" borderBottomWidth={1} flexShrink={0} gap="2" minH={HEADER_MIN_HEIGHT} px="3">
        <Text fontSize="sm" fontWeight="700">
          {t('nodes.nodePacks')}
        </Text>
        <Text color="fg.subtle" fontSize="xs">
          {nodePacks.length}
        </Text>
        <Box ms="auto">
          <ReloadNodesButton />
        </Box>
      </HStack>

      <NodePackList
        activePackName={activePackName}
        error={error}
        packs={nodePacks}
        searchTerm={searchTerm}
        status={status}
        onSearchChange={(value) => updateNodesUi({ searchTerm: value })}
        onSelect={openNodePackDetail}
        onUninstalled={(packName) => {
          if (activePackName === packName) {
            updateNodesUi({ activePackName: null });
          }
        }}
      />
    </Flex>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
