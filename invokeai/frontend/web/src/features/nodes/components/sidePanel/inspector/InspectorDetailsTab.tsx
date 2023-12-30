import { Box, Flex, HStack } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvText } from 'common/components/InvText/wrapper';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import NotesTextarea from 'features/nodes/components/flow/nodes/Invocation/NotesTextarea';
import { useNodeNeedsUpdate } from 'features/nodes/hooks/useNodeNeedsUpdate';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import EditableNodeTitle from './details/EditableNodeTitle';

const selector = createMemoizedSelector(stateSelector, ({ nodes }) => {
  const lastSelectedNodeId =
    nodes.selectedNodes[nodes.selectedNodes.length - 1];

  const lastSelectedNode = nodes.nodes.find(
    (node) => node.id === lastSelectedNodeId
  );

  const lastSelectedNodeTemplate = lastSelectedNode
    ? nodes.nodeTemplates[lastSelectedNode.data.type]
    : undefined;

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
    return (
      <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />
    );
  }

  return (
    <Content
      nodeId={data.nodeId}
      nodeVersion={data.nodeVersion}
      templateTitle={data.templateTitle}
    />
  );
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
        <Flex
          flexDir="column"
          position="relative"
          w="full"
          h="full"
          p={1}
          gap={2}
        >
          <EditableNodeTitle nodeId={props.nodeId} />
          <HStack>
            <InvControl label={t('nodes.nodeType')}>
              <InvText fontSize="sm" fontWeight="semibold">
                {props.templateTitle}
              </InvText>
            </InvControl>
            <InvControl label={t('nodes.nodeVersion')} isInvalid={needsUpdate}>
              <InvText fontSize="sm" fontWeight="semibold">
                {props.nodeVersion}
              </InvText>
            </InvControl>
          </HStack>
          <NotesTextarea nodeId={props.nodeId} />
        </Flex>
      </ScrollableContent>
    </Box>
  );
});

Content.displayName = 'Content';
