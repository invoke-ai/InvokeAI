import { Box, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { ImageOutput } from 'services/api/types';
import type { AnyResult } from 'services/events/types';

import ImageOutputPreview from './outputs/ImageOutputPreview';

const selector = createMemoizedSelector(selectNodesSlice, (nodes) => {
  const lastSelectedNodeId = nodes.selectedNodes[nodes.selectedNodes.length - 1];

  const lastSelectedNode = nodes.nodes.find((node) => node.id === lastSelectedNodeId);

  const lastSelectedNodeTemplate = lastSelectedNode ? nodes.templates[lastSelectedNode.data.type] : undefined;

  const nes = nodes.nodeExecutionStates[lastSelectedNodeId ?? '__UNKNOWN_NODE__'];

  if (!isInvocationNode(lastSelectedNode) || !nes || !lastSelectedNodeTemplate) {
    return;
  }

  return {
    outputs: nes.outputs,
    outputType: lastSelectedNodeTemplate.outputType,
  };
});

const InspectorOutputsTab = () => {
  const data = useAppSelector(selector);
  const { t } = useTranslation();

  if (!data) {
    return <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />;
  }

  if (data.outputs.length === 0) {
    return <IAINoContentFallback label={t('nodes.noOutputRecorded')} icon={null} />;
  }

  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <Flex position="relative" flexDir="column" alignItems="flex-start" p={1} gap={2} h="full" w="full">
          {data.outputType === 'image_output' ? (
            data.outputs.map((result, i) => (
              <ImageOutputPreview key={getKey(result, i)} output={result as ImageOutput} />
            ))
          ) : (
            <DataViewer data={data.outputs} label={t('nodes.nodeOutputs')} />
          )}
        </Flex>
      </ScrollableContent>
    </Box>
  );
};

export default memo(InspectorOutputsTab);

const getKey = (result: AnyResult, i: number) => `${result.type}-${i}`;
