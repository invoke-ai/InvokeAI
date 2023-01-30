import { Flex } from '@chakra-ui/react';

import IAICanvasUndoButton from 'features/canvas/components/IAICanvasToolbar/IAICanvasUndoButton';
import IAICanvasRedoButton from 'features/canvas/components/IAICanvasToolbar/IAICanvasRedoButton';
import UnifiedCanvasLayerSelect from './UnifiedCanvasToolbar/UnifiedCanvasLayerSelect';
import UnifiedCanvasToolSelect from './UnifiedCanvasToolbar/UnifiedCanvasToolSelect';
import UnifiedCanvasSettings from './UnifiedCanvasToolSettings/UnifiedCanvasSettings';
import UnifiedCanvasMoveTool from './UnifiedCanvasToolbar/UnifiedCanvasMoveTool';
import UnifiedCanvasResetView from './UnifiedCanvasToolbar/UnifiedCanvasResetView';
import UnifiedCanvasMergeVisible from './UnifiedCanvasToolbar/UnifiedCanvasMergeVisible';
import UnifiedCanvasSaveToGallery from './UnifiedCanvasToolbar/UnifiedCanvasSaveToGallery';
import UnifiedCanvasCopyToClipboard from './UnifiedCanvasToolbar/UnifiedCanvasCopyToClipboard';
import UnifiedCanvasDownloadImage from './UnifiedCanvasToolbar/UnifiedCanvasDownloadImage';
import UnifiedCanvasFileUploader from './UnifiedCanvasToolbar/UnifiedCanvasFileUploader';
import UnifiedCanvasResetCanvas from './UnifiedCanvasToolbar/UnifiedCanvasResetCanvas';
import UnifiedCanvasProcessingButtons from './UnifiedCanvasToolbar/UnifiedCanvasProcessingButtons';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';

const UnifiedCanvasToolbarBeta = () => {
  const shouldShowOptionsPanel = useAppSelector(
    (state: RootState) => state.options.shouldShowOptionsPanel
  );

  return (
    <Flex flexDirection={'column'} rowGap="0.5rem" width="6rem">
      <UnifiedCanvasLayerSelect />
      <UnifiedCanvasToolSelect />

      <Flex gap={'0.5rem'}>
        <UnifiedCanvasMoveTool />
        <UnifiedCanvasResetView />
      </Flex>

      <Flex columnGap={'0.5rem'}>
        <UnifiedCanvasMergeVisible />
        <UnifiedCanvasSaveToGallery />
      </Flex>
      <Flex columnGap={'0.5rem'}>
        <UnifiedCanvasCopyToClipboard />
        <UnifiedCanvasDownloadImage />
      </Flex>

      <Flex gap={'0.5rem'}>
        <IAICanvasUndoButton />
        <IAICanvasRedoButton />
      </Flex>

      <Flex gap={'0.5rem'}>
        <UnifiedCanvasFileUploader />
        <UnifiedCanvasResetCanvas />
      </Flex>

      <UnifiedCanvasSettings />
      {!shouldShowOptionsPanel && <UnifiedCanvasProcessingButtons />}
    </Flex>
  );
};

export default UnifiedCanvasToolbarBeta;
