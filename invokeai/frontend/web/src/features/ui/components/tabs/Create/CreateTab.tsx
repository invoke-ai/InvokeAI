import { Portal, TabPanel } from '@chakra-ui/react';
import { memo } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import GenerateParameters from './GenerateParameters';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import ImageGalleryContent from 'features/gallery/components/ImageGalleryContent';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import CreateTabContent from './GenerateContent';
import ResizeHandle from '../ResizeHandle';
import AnimatedImageToImagePanel from 'features/parameters/components/AnimatedImageToImagePanel';
import ImageToImageSettings from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageSettings';

const selector = createSelector(uiSelector, (ui) => {
  const {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
  } = ui;

  return {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
  };
});

const CreateTab = () => {
  const dispatch = useAppDispatch();
  const {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
  } = useAppSelector(selector);

  return (
    <PanelGroup
      direction="horizontal"
      style={{ height: '100%', width: '100%' }}
    >
      {shouldPinParametersPanel && shouldShowParametersPanel && (
        <>
          <Panel
            order={0}
            defaultSize={30}
            minSize={20}
            style={{ position: 'relative' }}
          >
            <GenerateParameters />
            <PinParametersPanelButton
              sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}
            />
          </Panel>
          <ResizeHandle />
        </>
      )}
      {shouldPinParametersPanel && shouldShowParametersPanel && (
        <>
          <Panel
            order={0}
            defaultSize={30}
            minSize={20}
            style={{ position: 'relative' }}
          >
            <ImageToImageSettings />
          </Panel>
          <ResizeHandle />
        </>
      )}
      <Panel
        order={1}
        minSize={30}
        onResize={() => {
          dispatch(requestCanvasRescale());
        }}
      >
        <CreateTabContent />
      </Panel>
      {shouldPinGallery && shouldShowGallery && (
        <>
          <ResizeHandle />
          <Panel order={2} defaultSize={10} minSize={10}>
            <ImageGalleryContent />
          </Panel>
        </>
      )}
    </PanelGroup>
  );
};

export default memo(CreateTab);
