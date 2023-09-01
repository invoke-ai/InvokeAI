import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { isInvocationNode } from 'features/nodes/types/types';
import { memo } from 'react';
import { ImageOutput } from 'services/api/types';
import { AnyResult } from 'services/events/types';
import ScrollableContent from '../ScrollableContent';
import ImageOutputPreview from './outputs/ImageOutputPreview';

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

    const nes =
      nodes.nodeExecutionStates[lastSelectedNodeId ?? '__UNKNOWN_NODE__'];

    return {
      node: lastSelectedNode,
      template: lastSelectedNodeTemplate,
      nes,
    };
  },
  defaultSelectorOptions
);

const InspectorOutputsTab = () => {
  const { node, template, nes } = useAppSelector(selector);

  if (!node || !nes || !isInvocationNode(node)) {
    return <IAINoContentFallback label="No node selected" icon={null} />;
  }

  if (nes.outputs.length === 0) {
    return <IAINoContentFallback label="No outputs recorded" icon={null} />;
  }

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
            position: 'relative',
            flexDir: 'column',
            alignItems: 'flex-start',
            p: 1,
            gap: 2,
            h: 'full',
            w: 'full',
          }}
        >
          {template?.outputType === 'image_output' ? (
            nes.outputs.map((result, i) => (
              <ImageOutputPreview
                key={getKey(result, i)}
                output={result as ImageOutput}
              />
            ))
          ) : (
            <DataViewer data={nes.outputs} label="Node Outputs" />
          )}
        </Flex>
      </ScrollableContent>
    </Box>
  );
};

export default memo(InspectorOutputsTab);

const getKey = (result: AnyResult, i: number) => `${result.type}-${i}`;
