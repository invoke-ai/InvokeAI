import { Box, Flex, HStack } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvText } from 'common/components/InvText/wrapper';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import NotesTextarea from 'features/nodes/components/flow/nodes/Invocation/NotesTextarea';
import type {
  InvocationNode,
  InvocationTemplate,
} from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { getNeedsUpdate } from 'features/nodes/util/node/nodeUpdate';
import { memo, useMemo } from 'react';
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

  return {
    node: lastSelectedNode,
    template: lastSelectedNodeTemplate,
  };
});

const InspectorDetailsTab = () => {
  const { node, template } = useAppSelector(selector);
  const { t } = useTranslation();

  if (!template || !isInvocationNode(node)) {
    return (
      <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />
    );
  }

  return <Content node={node} template={template} />;
};

export default memo(InspectorDetailsTab);

type ContentProps = {
  node: InvocationNode;
  template: InvocationTemplate;
};

const Content = memo(({ node, template }: ContentProps) => {
  const { t } = useTranslation();
  const needsUpdate = useMemo(
    () => getNeedsUpdate(node, template),
    [node, template]
  );
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
          <EditableNodeTitle nodeId={node.data.id} />
          <HStack>
            <InvControl label={t('nodes.nodeType')}>
              <InvText fontSize="sm" fontWeight="semibold">
                {template.title}
              </InvText>
            </InvControl>
            <InvControl label={t('nodes.nodeVersion')} isInvalid={needsUpdate}>
              <InvText fontSize="sm" fontWeight="semibold">
                {node.data.version}
              </InvText>
            </InvControl>
          </HStack>
          <NotesTextarea nodeId={node.data.id} />
        </Flex>
      </ScrollableContent>
    </Box>
  );
});

Content.displayName = 'Content';
