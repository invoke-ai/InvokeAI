import { Portal, TabPanel } from '@chakra-ui/react';
import { memo } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import CreateBaseSettings from './CreateBaseSettings';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import ImageGalleryContent from 'features/gallery/components/ImageGalleryContent';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import CreateTabContent from './CreateContent';
import ResizeHandle from '../ResizeHandle';
import AnimatedImageToImagePanel from 'features/parameters/components/AnimatedImageToImagePanel';
import ImageToImageSettings from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageSettings';
import CreateSidePanelPinned from './CreateSidePanelPinned';
import CreateTextParameters from './CreateBaseSettings';
import CreateImageSettings from './CreateImageSettings';

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

const CreateTab = () => {
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
      autoSaveId="createTab_pinned"
      direction="horizontal"
      style={{ height: '100%', width: '100%' }}
    >
      {shouldPinParametersPanel && shouldShowParametersPanel && (
        <>
          <Panel
            id="createTab_textParameters"
            order={0}
            defaultSize={25}
            minSize={25}
            style={{ position: 'relative' }}
          >
            <CreateTextParameters />
            <PinParametersPanelButton
              sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}
            />
          </Panel>
          {shouldShowImageParameters && (
            <>
              <ResizeHandle />
              <Panel
                id="createTab_imageParameters"
                order={1}
                defaultSize={25}
                minSize={25}
                style={{ position: 'relative' }}
              >
                <CreateImageSettings />
              </Panel>
            </>
          )}
          <ResizeHandle />
        </>
      )}
      <Panel
        id="createTab_content"
        order={2}
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
          <Panel id="createTab_gallery" order={3} defaultSize={10} minSize={10}>
            <ImageGalleryContent />
          </Panel>
        </>
      )}
    </PanelGroup>
  );
};

export default memo(CreateTab);
