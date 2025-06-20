import { Flex } from '@invoke-ai/ui-library';
import { ProgressImage } from 'features/gallery/components/ImageViewer/ProgressImage';
import { memo } from 'react';

export const GenerationProgressPanel = memo(() => (
  <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2}>
    <ProgressImage />
  </Flex>
));
GenerationProgressPanel.displayName = 'GenerationProgressPanel';
