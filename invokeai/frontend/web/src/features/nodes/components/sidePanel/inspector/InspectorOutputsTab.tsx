import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo } from 'react';
import ImageOutputPreview from './outputs/ImageOutputPreview';
import ScrollableContent from '../ScrollableContent';
import { AnyResult } from 'services/events/types';
import StringOutputPreview from './outputs/StringOutputPreview';
import NumberOutputPreview from './outputs/NumberOutputPreview';

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
    const lastSelectedNodeId =
      nodes.selectedNodes[nodes.selectedNodes.length - 1];

    const lastSelectedNode = nodes.nodes.find(
      (node) => node.id === lastSelectedNodeId
    );

    const nes =
      nodes.nodeExecutionStates[lastSelectedNodeId ?? '__UNKNOWN_NODE__'];

    return {
      node: lastSelectedNode,
      nes,
    };
  },
  defaultSelectorOptions
);

const InspectorOutputsTab = () => {
  const { node, nes } = useAppSelector(selector);

  if (!node || !nes) {
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
          {nes.outputs.map((result, i) => {
            if (result.type === 'string_output') {
              return (
                <StringOutputPreview key={getKey(result, i)} output={result} />
              );
            }
            if (result.type === 'float_output') {
              return (
                <NumberOutputPreview key={getKey(result, i)} output={result} />
              );
            }
            if (result.type === 'integer_output') {
              return (
                <NumberOutputPreview key={getKey(result, i)} output={result} />
              );
            }
            if (result.type === 'image_output') {
              return (
                <ImageOutputPreview key={getKey(result, i)} output={result} />
              );
            }
            return (
              <pre key={getKey(result, i)}>
                {JSON.stringify(result, null, 2)}
              </pre>
            );
          })}
        </Flex>
      </ScrollableContent>
    </Box>
  );
};

export default memo(InspectorOutputsTab);

const getKey = (result: AnyResult, i: number) => `${result.type}-${i}`;
