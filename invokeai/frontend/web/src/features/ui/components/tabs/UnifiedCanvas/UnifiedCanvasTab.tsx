import { Box, Flex, TabPanel } from '@chakra-ui/react';
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
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';

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
    <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
      {shouldPinParametersPanel && shouldShowParametersPanel && (
        <ParametersPinnedWrapper>
          <UnifiedCanvasParameters />
        </ParametersPinnedWrapper>
      )}
      {shouldUseCanvasBetaLayout ? (
        <UnifiedCanvasContentBeta />
      ) : (
        <UnifiedCanvasContent />
      )}
    </Flex>
  );
};

export default memo(UnifiedCanvasTab);
