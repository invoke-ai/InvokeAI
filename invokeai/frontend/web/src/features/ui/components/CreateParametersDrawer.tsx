import { isEqual } from 'lodash-es';
import ResizableDrawer from './common/ResizableDrawer/ResizableDrawer';
import TextTabParameters from './tabs/text/TextTabParameters';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector, uiSelector } from '../store/uiSelectors';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  setShouldShowParametersPanel,
  toggleParametersPanel,
} from '../store/uiSlice';
import { memo } from 'react';
import { Flex } from '@chakra-ui/react';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import PinParametersPanelButton from './PinParametersPanelButton';
import { Panel, PanelGroup } from 'react-resizable-panels';
import CreateSidePanelPinned from './tabs/text/TextTabSettingsPinned';
import CreateTextParameters from './tabs/text/TextTabParameters';
import ResizeHandle from './tabs/ResizeHandle';
import CreateImageSettings from './tabs/image/ImageTabSettings';

const selector = createSelector(
  [uiSelector, activeTabNameSelector, lightboxSelector],
  (ui, activeTabName, lightbox) => {
    const {
      shouldPinParametersPanel,
      shouldShowParametersPanel,
      shouldShowImageParameters,
    } = ui;
    const { isLightboxOpen } = lightbox;

    return {
      shouldPinParametersPanel,
      shouldShowParametersPanel,
      shouldShowImageParameters,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const CreateParametersPanel = () => {
  const dispatch = useAppDispatch();
  const {
    shouldPinParametersPanel,
    shouldShowParametersPanel,
    shouldShowImageParameters,
  } = useAppSelector(selector);

  const handleClosePanel = () => {
    dispatch(setShouldShowParametersPanel(false));
  };

  if (shouldPinParametersPanel) {
    return null;
  }

  return (
    <ResizableDrawer
      direction="left"
      isResizable={true}
      isOpen={shouldShowParametersPanel}
      onClose={handleClosePanel}
      minWidth={500}
    >
      <Flex flexDir="column" position="relative" h="full" w="full">
        <Flex
          paddingTop={1.5}
          paddingBottom={4}
          justifyContent="space-between"
          alignItems="center"
        >
          <InvokeAILogoComponent />
          <PinParametersPanelButton />
        </Flex>
        <PanelGroup
          autoSaveId="createTab_floatingParameters"
          direction="horizontal"
          style={{ height: '100%', width: '100%' }}
        >
          <>
            <Panel
              id="createTab_textParameters"
              order={0}
              defaultSize={25}
              minSize={25}
              style={{ position: 'relative' }}
            >
              <CreateTextParameters />
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
          </>
        </PanelGroup>
      </Flex>
    </ResizableDrawer>
  );
};

export default memo(CreateParametersPanel);
