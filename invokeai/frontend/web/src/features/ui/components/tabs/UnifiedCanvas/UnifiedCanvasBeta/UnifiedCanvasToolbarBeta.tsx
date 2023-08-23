import { Flex } from '@chakra-ui/react';

import IAICanvasRedoButton from 'features/canvas/components/IAICanvasToolbar/IAICanvasRedoButton';
import IAICanvasUndoButton from 'features/canvas/components/IAICanvasToolbar/IAICanvasUndoButton';
import { memo } from 'react';
import UnifiedCanvasSettings from './UnifiedCanvasToolSettings/UnifiedCanvasSettings';
import UnifiedCanvasCopyToClipboard from './UnifiedCanvasToolbar/UnifiedCanvasCopyToClipboard';
import UnifiedCanvasDownloadImage from './UnifiedCanvasToolbar/UnifiedCanvasDownloadImage';
import UnifiedCanvasFileUploader from './UnifiedCanvasToolbar/UnifiedCanvasFileUploader';
import UnifiedCanvasLayerSelect from './UnifiedCanvasToolbar/UnifiedCanvasLayerSelect';
import UnifiedCanvasMergeVisible from './UnifiedCanvasToolbar/UnifiedCanvasMergeVisible';
import UnifiedCanvasMoveTool from './UnifiedCanvasToolbar/UnifiedCanvasMoveTool';
import UnifiedCanvasResetCanvas from './UnifiedCanvasToolbar/UnifiedCanvasResetCanvas';
import UnifiedCanvasResetView from './UnifiedCanvasToolbar/UnifiedCanvasResetView';
import UnifiedCanvasSaveToGallery from './UnifiedCanvasToolbar/UnifiedCanvasSaveToGallery';
import UnifiedCanvasToolSelect from './UnifiedCanvasToolbar/UnifiedCanvasToolSelect';

const UnifiedCanvasToolbarBeta = () => {
  return (
    <Flex flexDirection="column" rowGap={2} width="min-content">
      <UnifiedCanvasLayerSelect />
      <UnifiedCanvasToolSelect />

      <Flex gap={2}>
        <UnifiedCanvasMoveTool />
        <UnifiedCanvasResetView />
      </Flex>

      <Flex columnGap={2}>
        <UnifiedCanvasMergeVisible />
        <UnifiedCanvasSaveToGallery />
      </Flex>
      <Flex columnGap={2}>
        <UnifiedCanvasCopyToClipboard />
        <UnifiedCanvasDownloadImage />
      </Flex>

      <Flex gap={2}>
        <IAICanvasUndoButton />
        <IAICanvasRedoButton />
      </Flex>

      <Flex gap={2}>
        <UnifiedCanvasFileUploader />
        <UnifiedCanvasResetCanvas />
      </Flex>

      <UnifiedCanvasSettings />
    </Flex>
  );
};

export default memo(UnifiedCanvasToolbarBeta);
