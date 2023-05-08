import { Portal, TabPanel } from '@chakra-ui/react';
import { memo } from 'react';
import { Panel, PanelGroup } from 'react-resizable-panels';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import ResizeHandle from '../ResizeHandle';
import ImageTabParameters from './ImageTabParameters';
import ImageTabImageParameters from './ImageTabImageParameters';
import TextTabMain from '../text/TextTabMain';
import InitialImagePreview from 'features/parameters/components/AdvancedParameters/ImageToImage/InitialImagePreview';
import InitialImageDisplay from 'features/parameters/components/AdvancedParameters/ImageToImage/InitialImageDisplay';

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
  const {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
    shouldShowImageParameters,
  } = useAppSelector(selector);

  return (
    <PanelGroup
      autoSaveId="imageTab"
      direction="horizontal"
      style={{ height: '100%', width: '100%' }}
    >
      {shouldPinParametersPanel && shouldShowParametersPanel && (
        <>
          <Panel
            id="imageTab_parameters"
            order={0}
            defaultSize={25}
            minSize={25}
            style={{ position: 'relative' }}
          >
            <ImageTabParameters />
            <PinParametersPanelButton
              sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}
            />
          </Panel>
          <ResizeHandle />
        </>
      )}
      <Panel id="imageTab_content" order={1}>
        <PanelGroup
          autoSaveId="imageTab_contentWrapper"
          direction="horizontal"
          style={{ height: '100%', width: '100%' }}
        >
          <Panel
            id="imageTab_initImage"
            order={0}
            defaultSize={50}
            minSize={25}
            style={{ position: 'relative' }}
          >
            <InitialImageDisplay />
          </Panel>
          <ResizeHandle />
          <Panel
            id="imageTab_selectedImage"
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
      </Panel>
    </PanelGroup>
  );
};

export default memo(TextTab);
