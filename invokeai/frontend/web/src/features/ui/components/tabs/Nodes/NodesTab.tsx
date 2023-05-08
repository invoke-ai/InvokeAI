import { TabPanel } from '@chakra-ui/react';
import { memo } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import ImageGalleryContent from 'features/gallery/components/ImageGalleryContent';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import ResizeHandle from '../ResizeHandle';
import NodeEditor from 'features/nodes/components/NodeEditor';

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

const NodesTab = () => {
  const dispatch = useAppDispatch();
  const {
    shouldPinGallery,
    shouldShowGallery,
    shouldPinParametersPanel,
    shouldShowParametersPanel,
  } = useAppSelector(selector);

  return (
    <PanelGroup
      autoSaveId="nodesTab"
      direction="horizontal"
      style={{ height: '100%', width: '100%' }}
    >
      <Panel
        order={1}
        minSize={30}
        onResize={() => {
          dispatch(requestCanvasRescale());
        }}
      >
        <NodeEditor />
      </Panel>
      {shouldPinGallery && shouldShowGallery && (
        <>
          <ResizeHandle />
          <Panel order={2} defaultSize={10} minSize={10} collapsible={true}>
            <ImageGalleryContent />
          </Panel>
        </>
      )}
    </PanelGroup>
  );
};

export default memo(NodesTab);
