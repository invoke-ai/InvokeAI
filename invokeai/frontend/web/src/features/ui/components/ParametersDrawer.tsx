import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { setShouldShowParametersPanel } from 'features/ui/store/uiSlice';
import { memo, useMemo } from 'react';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';
import PinParametersPanelButton from './PinParametersPanelButton';
import OverlayScrollable from './common/OverlayScrollable';
import ResizableDrawer from './common/ResizableDrawer/ResizableDrawer';
import ImageToImageTabParameters from './tabs/ImageToImage/ImageToImageTabParameters';
import TextToImageTabParameters from './tabs/TextToImage/TextToImageTabParameters';
import UnifiedCanvasParameters from './tabs/UnifiedCanvas/UnifiedCanvasParameters';

const selector = createSelector(
  [uiSelector, activeTabNameSelector, lightboxSelector],
  (ui, activeTabName, lightbox) => {
    const { shouldPinParametersPanel, shouldShowParametersPanel } = ui;

    const { isLightboxOpen } = lightbox;

    return {
      activeTabName,
      shouldPinParametersPanel,
      shouldShowParametersPanel,
    };
  },
  defaultSelectorOptions
);

const ParametersDrawer = () => {
  const dispatch = useAppDispatch();
  const { shouldPinParametersPanel, shouldShowParametersPanel, activeTabName } =
    useAppSelector(selector);

  const handleClosePanel = () => {
    dispatch(setShouldShowParametersPanel(false));
  };

  const drawerContent = useMemo(() => {
    if (activeTabName === 'txt2img') {
      return <TextToImageTabParameters />;
    }

    if (activeTabName === 'img2img') {
      return <ImageToImageTabParameters />;
    }

    if (activeTabName === 'unifiedCanvas') {
      return <UnifiedCanvasParameters />;
    }

    return null;
  }, [activeTabName]);

  if (shouldPinParametersPanel) {
    return null;
  }

  return (
    <ResizableDrawer
      direction="left"
      isResizable={false}
      isOpen={shouldShowParametersPanel}
      onClose={handleClosePanel}
    >
      <Flex
        sx={{
          flexDir: 'column',
          h: 'full',
          w: PARAMETERS_PANEL_WIDTH,
          gap: 2,
          position: 'relative',
          flexShrink: 0,
          overflowY: 'auto',
        }}
      >
        <Flex
          paddingTop={1.5}
          paddingBottom={4}
          justifyContent="space-between"
          alignItems="center"
        >
          <InvokeAILogoComponent />
          <PinParametersPanelButton />
        </Flex>
        <Flex
          sx={{
            gap: 2,
            flexDirection: 'column',
            h: 'full',
            w: 'full',
          }}
        >
          {drawerContent}
        </Flex>
      </Flex>
    </ResizableDrawer>
  );
};

export default memo(ParametersDrawer);
