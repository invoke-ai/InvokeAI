import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { InvocationTemplate, NodeData } from 'features/nodes/types/types';
import { memo } from 'react';
import NotesTextarea from '../../flow/nodes/Invocation/NotesTextarea';
import NodeTitle from '../../flow/nodes/common/NodeTitle';
import ScrollableContent from '../ScrollableContent';

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
      data: lastSelectedNode?.data,
      template: lastSelectedNodeTemplate,
    };
  },
  defaultSelectorOptions
);

const InspectorDetailsTab = () => {
  const { data, template } = useAppSelector(selector);

  if (!template || !data) {
    return <IAINoContentFallback label="No node selected" icon={null} />;
  }

  return <Content data={data} template={template} />;
};

export default memo(InspectorDetailsTab);

const Content = (props: { data: NodeData; template: InvocationTemplate }) => {
  const { data } = props;

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
          <NodeTitle nodeId={data.id} />
          <NotesTextarea nodeId={data.id} />
        </Flex>
      </ScrollableContent>
    </Box>
  );
};
