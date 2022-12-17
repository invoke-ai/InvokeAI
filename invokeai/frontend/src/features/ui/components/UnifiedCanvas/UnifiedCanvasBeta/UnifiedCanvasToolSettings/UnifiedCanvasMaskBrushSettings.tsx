import { Flex } from '@chakra-ui/react';
import UnifiedCanvasBrushSettings from './UnifiedCanvasBrushSettings';
import UnifiedCanvasClearMask from './UnifiedCanvasClearMask';
import UnifiedCanvasEnableMask from './UnifiedCanvasEnableMask';
import UnifiedCanvasPreserveMask from './UnifiedCanvasPreserveMask';

export default function UnifiedCanvasMaskBrushSettings() {
  return (
    <Flex gap="1rem" alignItems="center">
      <UnifiedCanvasBrushSettings />
      <UnifiedCanvasEnableMask />
      <UnifiedCanvasPreserveMask />
      <UnifiedCanvasClearMask />
    </Flex>
  );
}
