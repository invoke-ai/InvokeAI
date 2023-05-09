import { Box, Flex, Portal, TabPanel } from '@chakra-ui/react';
import { memo, useCallback, useRef } from 'react';
import {
  ImperativePanelHandle,
  Panel,
  PanelGroup,
} from 'react-resizable-panels';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import ResizeHandle from '../ResizeHandle';
import ImageTabParameters from './ImageTabParameters';
import ImageTabImageParameters from './ImageTabImageParameters';
import TextTabMain from '../text/TextTabMain';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';
import { ImperativePanelGroupHandle } from 'react-resizable-panels';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';
import InitialImageDisplay from 'features/parameters/components/Parameters/ImageToImage/InitialImageDisplay';

const selector = createSelector(uiSelector, (ui) => {
  const {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
    shouldShowImageParameters,
  } = ui;

  return {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
    shouldShowImageParameters,
  };
});

const TextTab = () => {
  const dispatch = useAppDispatch();
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);

  const handleDoubleClickHandle = useCallback(() => {
    if (!panelGroupRef.current) {
      return;
    }

    panelGroupRef.current.setLayout([50, 50]);
  }, []);

  const {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
    shouldShowImageParameters,
  } = useAppSelector(selector);

  return (
    <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
      {shouldPinParametersPanel && shouldShowParametersPanel && (
        <ParametersPinnedWrapper>
          <ImageTabParameters />
        </ParametersPinnedWrapper>
      )}
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
            <TextTabMain />
          </Panel>
        </PanelGroup>
      </Box>
    </Flex>
  );
};

export default memo(TextTab);
