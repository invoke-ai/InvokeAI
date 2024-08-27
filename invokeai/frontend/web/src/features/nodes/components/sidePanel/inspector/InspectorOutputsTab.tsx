import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { useExecutionState } from 'features/nodes/hooks/useExecutionState';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectLastSelectedNode, selectNodesSlice } from 'features/nodes/store/selectors';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyInvocationOutput, ImageOutput } from 'services/api/types';

import ImageOutputPreview from './outputs/ImageOutputPreview';

const InspectorOutputsTab = () => {
  const templates = useStore($templates);
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectNodesSlice, (nodes) => {
        const lastSelectedNode = selectLastSelectedNode(nodes);
        const lastSelectedNodeTemplate = lastSelectedNode ? templates[lastSelectedNode.data.type] : undefined;

        if (!isInvocationNode(lastSelectedNode) || !lastSelectedNodeTemplate) {
          return;
        }

        return {
          nodeId: lastSelectedNode.id,
          outputType: lastSelectedNodeTemplate.outputType,
        };
      }),
    [templates]
  );
  const data = useAppSelector(selector);
  const nes = useExecutionState(data?.nodeId);
  const { t } = useTranslation();

  if (!data || !nes) {
    return <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />;
  }

  if (nes.outputs.length === 0) {
    return <IAINoContentFallback label={t('nodes.noOutputRecorded')} icon={null} />;
  }

  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <Flex position="relative" flexDir="column" alignItems="flex-start" p={1} gap={2} h="full" w="full">
          {data.outputType === 'image_output' ? (
            nes.outputs.map((result, i) => (
              <ImageOutputPreview key={getKey(result, i)} output={result as ImageOutput} />
            ))
          ) : (
            <DataViewer data={nes.outputs} label={t('nodes.nodeOutputs')} />
          )}
        </Flex>
      </ScrollableContent>
    </Box>
  );
};

export default memo(InspectorOutputsTab);

const getKey = (result: AnyInvocationOutput, i: number) => `${result.type}-${i}`;
