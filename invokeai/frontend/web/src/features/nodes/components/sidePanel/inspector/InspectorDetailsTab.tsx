import { Box, Flex, FormControl, FormLabel, HStack, Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import NotesTextarea from 'features/nodes/components/flow/nodes/Invocation/NotesTextarea';
import { useNodeNeedsUpdate } from 'features/nodes/hooks/useNodeNeedsUpdate';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import EditableNodeTitle from './details/EditableNodeTitle';

const selector = createMemoizedSelector(selectNodesSlice, (nodes) => {
  const lastSelectedNodeId = nodes.selectedNodes[nodes.selectedNodes.length - 1];

  const lastSelectedNode = nodes.nodes.find((node) => node.id === lastSelectedNodeId);

  const lastSelectedNodeTemplate = lastSelectedNode ? nodes.templates[lastSelectedNode.data.type] : undefined;

  if (!isInvocationNode(lastSelectedNode) || !lastSelectedNodeTemplate) {
    return;
  }

  return {
    nodeId: lastSelectedNode.data.id,
    nodeVersion: lastSelectedNode.data.version,
    templateTitle: lastSelectedNodeTemplate.title,
  };
});

const InspectorDetailsTab = () => {
  const data = useAppSelector(selector);
  const { t } = useTranslation();

  if (!data) {
    return <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />;
  }

  return <Content nodeId={data.nodeId} nodeVersion={data.nodeVersion} templateTitle={data.templateTitle} />;
};

export default memo(InspectorDetailsTab);

type ContentProps = {
  nodeId: string;
  nodeVersion: string;
  templateTitle: string;
};

const Content = memo((props: ContentProps) => {
  const { t } = useTranslation();
  const needsUpdate = useNodeNeedsUpdate(props.nodeId);
  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <Flex flexDir="column" position="relative" w="full" h="full" p={1} gap={2}>
          <EditableNodeTitle nodeId={props.nodeId} />
          <HStack>
            <FormControl>
              <FormLabel>{t('nodes.nodeType')}</FormLabel>
              <Text fontSize="sm" fontWeight="semibold">
                {props.templateTitle}
              </Text>
            </FormControl>
            <FormControl isInvalid={needsUpdate}>
              <FormLabel>{t('nodes.nodeVersion')}</FormLabel>
              <Text fontSize="sm" fontWeight="semibold">
                {props.nodeVersion}
              </Text>
            </FormControl>
          </HStack>
          <NotesTextarea nodeId={props.nodeId} />
        </Flex>
      </ScrollableContent>
    </Box>
  );
});

Content.displayName = 'Content';
