import { Box } from '@invoke-ai/ui-library';
import { ControlLayersEditor } from 'features/controlLayers/components/ControlLayersEditor';
import { memo } from 'react';

const TextToImageTab = () => {
  return (
    <Box layerStyle="first" position="relative" w="full" h="full" p={2} borderRadius="base">
      <ControlLayersEditor />
    </Box>
  );
};

export default memo(TextToImageTab);
