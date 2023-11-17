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
import ScrollableContent from '../ScrollableContent';
import InputFields from './details/InputFields';

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
            gap: 2,
            w: 'full',
          }}
        >
          <FormControl>
            <FormLabel>Type</FormLabel>
            <Text fontSize="sm" fontWeight={600}>
              {props.template.title} ({props.template.type})
            </Text>
          </FormControl>
          <FormControl>
            <FormLabel>Description</FormLabel>
            <Text fontSize="sm" fontWeight={600}>
              {props.template.description}
            </Text>
          </FormControl>
          <InputFields nodeId={props.node.id} />
        </Flex>
      </ScrollableContent>
    </Box>
  );
};
