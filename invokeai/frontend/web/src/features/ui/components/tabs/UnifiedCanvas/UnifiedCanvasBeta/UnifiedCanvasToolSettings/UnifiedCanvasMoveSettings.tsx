import { Flex } from '@chakra-ui/layout';
import UnifiedCanvasDarkenOutsideSelection from './UnifiedCanvasDarkenOutsideSelection';
import UnifiedCanvasShowGrid from './UnifiedCanvasShowGrid';
import UnifiedCanvasSnapToGrid from './UnifiedCanvasSnapToGrid';

export default function UnifiedCanvasMoveSettings() {
  return (
    <Flex alignItems="center" gap={4}>
      <UnifiedCanvasShowGrid />
      <UnifiedCanvasSnapToGrid />
      <UnifiedCanvasDarkenOutsideSelection />
    </Flex>
  );
}
