import { Box, Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import InitialImageDisplay from 'features/parameters/components/Parameters/ImageToImage/InitialImageDisplay';
import SDXLImageToImageTabParameters from 'features/sdxl/components/SDXLImageToImageTabParameters';
import { memo, useCallback, useRef } from 'react';
import {
  ImperativePanelGroupHandle,
  Panel,
  PanelGroup,
} from 'react-resizable-panels';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';
import ResizeHandle from '../ResizeHandle';
import TextToImageTabMain from '../TextToImage/TextToImageTabMain';
import ImageToImageTabParameters from './ImageToImageTabParameters';

const ImageToImageTab = () => {
  const dispatch = useAppDispatch();
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const model = useAppSelector((state: RootState) => state.generation.model);

  const handleDoubleClickHandle = useCallback(() => {
    if (!panelGroupRef.current) {
      return;
    }

    panelGroupRef.current.setLayout([50, 50]);
  }, []);

  return (
    <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
      <ParametersPinnedWrapper>
        {model && model.base_model === 'sdxl' ? (
          <SDXLImageToImageTabParameters />
        ) : (
          <ImageToImageTabParameters />
        )}
      </ParametersPinnedWrapper>
      <Box sx={{ w: 'full', h: 'full' }}>
        <PanelGroup
          ref={panelGroupRef}
          autoSaveId="imageTab.content"
          direction="horizontal"
          style={{ height: '100%', width: '100%' }}
        >
          <Panel
            id="imageTab.content.initImage"
            order={0}
            defaultSize={50}
            minSize={25}
            style={{ position: 'relative' }}
          >
            <InitialImageDisplay />
          </Panel>
          <ResizeHandle onDoubleClick={handleDoubleClickHandle} />
          <Panel
            id="imageTab.content.selectedImage"
            order={1}
            defaultSize={50}
            minSize={25}
            onResize={() => {
              dispatch(requestCanvasRescale());
            }}
          >
            <TextToImageTabMain />
          </Panel>
        </PanelGroup>
      </Box>
    </Flex>
  );
};

export default memo(ImageToImageTab);
