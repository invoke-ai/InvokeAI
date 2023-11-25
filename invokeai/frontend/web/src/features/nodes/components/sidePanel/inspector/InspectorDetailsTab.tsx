import {
  Box,
  Flex,
  FormControl,
  FormLabel,
  HStack,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIIconButton from 'common/components/IAIIconButton';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { useNodeVersion } from 'features/nodes/hooks/useNodeVersion';
import {
  InvocationNodeData,
  InvocationTemplate,
  isInvocationNode,
} from 'features/nodes/types/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSync } from 'react-icons/fa';
import { Node } from 'reactflow';
import NotesTextarea from '../../flow/nodes/Invocation/NotesTextarea';
import ScrollableContent from '../ScrollableContent';
import EditableNodeTitle from './details/EditableNodeTitle';

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
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
  },
  defaultSelectorOptions
);

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

const Content = (props: {
  node: Node<InvocationNodeData>;
  template: InvocationTemplate;
}) => {
  const { t } = useTranslation();
  const { needsUpdate, updateNode } = useNodeVersion(props.node.id);
  return (
    <Box
      sx={{
        position: 'relative',
        w: 'full',
        h: 'full',
      }}
    >
      <ScrollableContent>
        <Flex
          sx={{
            flexDir: 'column',
            position: 'relative',
            p: 1,
            gap: 2,
            w: 'full',
          }}
        >
          <EditableNodeTitle nodeId={props.node.data.id} />
          <HStack>
            <FormControl>
              <FormLabel>{t('nodes.nodeType')}</FormLabel>
              <Text fontSize="sm" fontWeight={600}>
                {props.template.title}
              </Text>
            </FormControl>
            <Flex
              flexDir="row"
              alignItems="center"
              justifyContent="space-between"
              w="full"
            >
              <FormControl isInvalid={needsUpdate}>
                <FormLabel>{t('nodes.nodeVersion')}</FormLabel>
                <Text fontSize="sm" fontWeight={600}>
                  {props.node.data.version}
                </Text>
              </FormControl>
              {needsUpdate && (
                <IAIIconButton
                  aria-label={t('nodes.updateNode')}
                  tooltip={t('nodes.updateNode')}
                  icon={<FaSync />}
                  onClick={updateNode}
                />
              )}
            </Flex>
          </HStack>
          <NotesTextarea nodeId={props.node.data.id} />
        </Flex>
      </ScrollableContent>
    </Box>
  );
};
