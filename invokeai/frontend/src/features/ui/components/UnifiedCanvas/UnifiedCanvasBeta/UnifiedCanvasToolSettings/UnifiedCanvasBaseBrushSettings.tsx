import { Flex } from '@chakra-ui/react';
import UnifiedCanvasBrushSettings from './UnifiedCanvasBrushSettings';
import UnifiedCanvasLimitStrokesToBox from './UnifiedCanvasLimitStrokesToBox';

export default function UnifiedCanvasBaseBrushSettings() {
  return (
    <Flex gap="1rem" alignItems="center">
      <UnifiedCanvasBrushSettings />
      <UnifiedCanvasLimitStrokesToBox />
    </Flex>
  );
}
