import { Box } from '@invoke-ai/ui-library';
import { Viewer } from 'features/viewer/components/Viewer';
import { memo } from 'react';

const TextToImageTab = () => {
  return (
    <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
      <Viewer />
    </Box>
  );
};

export default memo(TextToImageTab);
