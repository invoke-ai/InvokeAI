import { Box, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { TemplateGate } from 'features/nodes/components/sidePanel/inspector/NodeTemplateGate';
import { useNodeExecutionState } from 'features/nodes/hooks/useNodeExecutionState';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import { selectLastSelectedNodeId } from 'features/nodes/store/selectors';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyInvocationOutput, ImageOutput } from 'services/api/types';

import ImageOutputPreview from './ImageOutputPreview';

const InspectorOutputsTab = () => {
  const lastSelectedNodeId = useAppSelector(selectLastSelectedNodeId);
  const { t } = useTranslation();

  if (!lastSelectedNodeId) {
    return <IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />;
  }

  return (
    <TemplateGate
      nodeId={lastSelectedNodeId}
      fallback={<IAINoContentFallback label={t('nodes.noNodeSelected')} icon={null} />}
    >
      <Content nodeId={lastSelectedNodeId} />
    </TemplateGate>
  );
};

export default memo(InspectorOutputsTab);

const getKey = (result: AnyInvocationOutput, i: number) => `${result.type}-${i}`;

const Content = memo(({ nodeId }: { nodeId: string }) => {
  const { t } = useTranslation();
  const template = useNodeTemplate(nodeId);
  const nes = useNodeExecutionState(nodeId);

  if (!nes || nes.outputs.length === 0) {
    return <IAINoContentFallback label={t('nodes.noOutputRecorded')} icon={null} />;
  }

  return (
    <Box position="relative" w="full" h="full">
      <ScrollableContent>
        <Flex position="relative" flexDir="column" alignItems="flex-start" p={1} gap={2} h="full" w="full">
          {template.outputType === 'image_output' ? (
            nes.outputs.map((result, i) => (
              <ImageOutputPreview key={getKey(result, i)} output={result as ImageOutput} />
            ))
          ) : (
            <DataViewer data={nes.outputs} label={t('nodes.nodeOutputs')} bg="base.850" color="base.200" />
          )}
        </Flex>
      </ScrollableContent>
    </Box>
  );
});
Content.displayName = 'Content';
