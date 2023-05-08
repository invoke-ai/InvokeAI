import { Portal, TabPanel } from '@chakra-ui/react';
import { memo } from 'react';
import { Panel, PanelGroup } from 'react-resizable-panels';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import TextTabMain from './TextTabMain';
import ResizeHandle from '../ResizeHandle';
import TextTabSettings from './TextTabParameters';

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
      autoSaveId="textTab"
      direction="horizontal"
      style={{ height: '100%', width: '100%' }}
    >
      {shouldPinParametersPanel && shouldShowParametersPanel && (
        <>
          <Panel
            id="textTab_settings"
            order={0}
            defaultSize={25}
            minSize={25}
            style={{ position: 'relative' }}
          >
            <TextTabSettings />
            <PinParametersPanelButton
              sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}
            />
          </Panel>
          <ResizeHandle />
        </>
      )}
      <Panel
        id="textTab_main"
        order={2}
        minSize={30}
        onResize={() => {
          dispatch(requestCanvasRescale());
        }}
      >
        <TextTabMain />
      </Panel>
    </PanelGroup>
  );
};

export default memo(TextTab);
