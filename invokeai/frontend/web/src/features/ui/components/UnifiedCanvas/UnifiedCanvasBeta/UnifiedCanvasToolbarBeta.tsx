import { Flex } from '@chakra-ui/react';

import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import IAICanvasRedoButton from 'features/canvas/components/IAICanvasToolbar/IAICanvasRedoButton';
import IAICanvasUndoButton from 'features/canvas/components/IAICanvasToolbar/IAICanvasUndoButton';
import UnifiedCanvasCopyToClipboard from './UnifiedCanvasToolbar/UnifiedCanvasCopyToClipboard';
import UnifiedCanvasDownloadImage from './UnifiedCanvasToolbar/UnifiedCanvasDownloadImage';
import UnifiedCanvasFileUploader from './UnifiedCanvasToolbar/UnifiedCanvasFileUploader';
import UnifiedCanvasLayerSelect from './UnifiedCanvasToolbar/UnifiedCanvasLayerSelect';
import UnifiedCanvasMergeVisible from './UnifiedCanvasToolbar/UnifiedCanvasMergeVisible';
import UnifiedCanvasMoveTool from './UnifiedCanvasToolbar/UnifiedCanvasMoveTool';
import UnifiedCanvasProcessingButtons from './UnifiedCanvasToolbar/UnifiedCanvasProcessingButtons';
import UnifiedCanvasResetCanvas from './UnifiedCanvasToolbar/UnifiedCanvasResetCanvas';
import UnifiedCanvasResetView from './UnifiedCanvasToolbar/UnifiedCanvasResetView';
import UnifiedCanvasSaveToGallery from './UnifiedCanvasToolbar/UnifiedCanvasSaveToGallery';
import UnifiedCanvasToolSelect from './UnifiedCanvasToolbar/UnifiedCanvasToolSelect';
import UnifiedCanvasSettings from './UnifiedCanvasToolSettings/UnifiedCanvasSettings';

const UnifiedCanvasToolbarBeta = () => {
  const shouldShowParametersPanel = useAppSelector(
    (state: RootState) => state.ui.shouldShowParametersPanel
  );

  return (
    <Flex flexDirection="column" rowGap="0.5rem" width="6rem">
      <UnifiedCanvasLayerSelect />
      <UnifiedCanvasToolSelect />

      <Flex gap="0.5rem">
        <UnifiedCanvasMoveTool />
        <UnifiedCanvasResetView />
      </Flex>

      <Flex columnGap="0.5rem">
        <UnifiedCanvasMergeVisible />
        <UnifiedCanvasSaveToGallery />
      </Flex>
      <Flex columnGap="0.5rem">
        <UnifiedCanvasCopyToClipboard />
        <UnifiedCanvasDownloadImage />
      </Flex>

      <Flex gap="0.5rem">
        <IAICanvasUndoButton />
        <IAICanvasRedoButton />
      </Flex>

      <Flex gap="0.5rem">
        <UnifiedCanvasFileUploader />
        <UnifiedCanvasResetCanvas />
      </Flex>

      <UnifiedCanvasSettings />
      {!shouldShowParametersPanel && <UnifiedCanvasProcessingButtons />}
    </Flex>
  );
};

export default UnifiedCanvasToolbarBeta;
