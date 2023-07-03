import { Box, Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import InitialImageDisplay from 'features/parameters/components/Parameters/ImageToImage/InitialImageDisplay';
import { memo, useCallback, useRef } from 'react';
import {
  ImperativePanelGroupHandle,
  Panel,
  PanelGroup,
} from 'react-resizable-panels';
import ResizeHandle from '../ResizeHandle';
import TextToImageTabMain from '../TextToImage/TextToImageTabMain';
import BatchManager from 'features/batch/components/BatchManager';

const ImageToImageTab = () => {
  const dispatch = useAppDispatch();
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);

  const handleDoubleClickHandle = useCallback(() => {
    if (!panelGroupRef.current) {
      return;
    }

    panelGroupRef.current.setLayout([50, 50]);
  }, []);

  return (
    <Flex
      layerStyle={'first'}
      sx={{
        gap: 4,
        p: 4,
        w: 'full',
        h: 'full',
        borderRadius: 'base',
      }}
    >
      <BatchManager />
    </Flex>
  );
};

export default memo(ImageToImageTab);
