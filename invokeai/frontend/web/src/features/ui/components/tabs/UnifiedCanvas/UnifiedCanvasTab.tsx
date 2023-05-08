import { TabPanel } from '@chakra-ui/react';
import { memo } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import ImageGalleryContent from 'features/gallery/components/ImageGalleryContent';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import UnifiedCanvasContent from './UnifiedCanvasContent';
import ResizeHandle from '../ResizeHandle';
import UnifiedCanvasParameters from './UnifiedCanvasParameters';
import UnifiedCanvasContentBeta from './UnifiedCanvasBeta/UnifiedCanvasContentBeta';

const selector = createSelector(uiSelector, (ui) => {
  const {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
    shouldUseCanvasBetaLayout,
  } = ui;

  return {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
    shouldUseCanvasBetaLayout,
  };
});

const UnifiedCanvasTab = () => {
  const dispatch = useAppDispatch();
  const {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
    shouldUseCanvasBetaLayout,
  } = useAppSelector(selector);

  return (
    <PanelGroup
      autoSaveId="canvasTab"
      direction="horizontal"
      style={{ height: '100%', width: '100%' }}
    >
      {shouldPinParametersPanel && shouldShowParametersPanel && (
        <>
          <Panel
            id="canvasTab_parameters"
            order={0}
            defaultSize={30}
            minSize={20}
            style={{ position: 'relative' }}
          >
            <UnifiedCanvasParameters />
            <PinParametersPanelButton
              sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}
            />
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
        {shouldUseCanvasBetaLayout ? (
          <UnifiedCanvasContentBeta />
        ) : (
          <UnifiedCanvasContent />
        )}
      </Panel>
    </PanelGroup>
  );
};

export default memo(UnifiedCanvasTab);
