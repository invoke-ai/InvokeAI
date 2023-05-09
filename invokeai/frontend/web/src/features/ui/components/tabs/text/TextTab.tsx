import { Box, Flex, Portal, TabPanel } from '@chakra-ui/react';
import { memo } from 'react';
import { Panel, PanelGroup } from 'react-resizable-panels';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import TextTabMain from './TextTabMain';
import ResizeHandle from '../ResizeHandle';
import TextTabParameters from './TextTabParameters';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';

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
    <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
      {shouldPinParametersPanel && shouldShowParametersPanel && (
        <ParametersPinnedWrapper>
          <TextTabParameters />
        </ParametersPinnedWrapper>
      )}
      <TextTabMain />
    </Flex>
  );
};

export default memo(TextTab);
