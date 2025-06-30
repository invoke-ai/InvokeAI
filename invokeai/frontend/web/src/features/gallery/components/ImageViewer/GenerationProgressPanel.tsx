import { Flex } from '@invoke-ai/ui-library';
import { ProgressImage } from 'features/gallery/components/ImageViewer/ProgressImage';
import { memo } from 'react';

import { ProgressIndicator } from './ProgressIndicator';

export const GenerationProgressPanel = memo(() => (
  <Flex position="relative" flexDir="column" w="full" h="full" overflow="hidden" p={2}>
    <ProgressImage />
    <ProgressIndicator position="absolute" top={6} right={6} size={8} />
  </Flex>
));
GenerationProgressPanel.displayName = 'GenerationProgressPanel';
