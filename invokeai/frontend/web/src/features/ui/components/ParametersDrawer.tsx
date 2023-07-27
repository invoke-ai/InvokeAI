import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import SDXLImageToImageTabParameters from 'features/sdxl/components/SDXLImageToImageTabParameters';
import SDXLTextToImageTabParameters from 'features/sdxl/components/SDXLTextToImageTabParameters';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { setShouldShowParametersPanel } from 'features/ui/store/uiSlice';
import { memo, useMemo } from 'react';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';
import PinParametersPanelButton from './PinParametersPanelButton';
import ResizableDrawer from './common/ResizableDrawer/ResizableDrawer';
import ImageToImageTabParameters from './tabs/ImageToImage/ImageToImageTabParameters';
import TextToImageTabParameters from './tabs/TextToImage/TextToImageTabParameters';
import UnifiedCanvasParameters from './tabs/UnifiedCanvas/UnifiedCanvasParameters';

const selector = createSelector(
  [uiSelector, activeTabNameSelector],
  (ui, activeTabName) => {
    const { shouldPinParametersPanel, shouldShowParametersPanel } = ui;

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

  const model = useAppSelector((state: RootState) => state.generation.model);

  const drawerContent = useMemo(() => {
    if (activeTabName === 'txt2img') {
      return model && model.base_model === 'sdxl' ? (
        <SDXLTextToImageTabParameters />
      ) : (
        <TextToImageTabParameters />
      );
    }

    if (activeTabName === 'img2img') {
      return model && model.base_model === 'sdxl' ? (
        <SDXLImageToImageTabParameters />
      ) : (
        <ImageToImageTabParameters />
      );
    }

    if (activeTabName === 'unifiedCanvas') {
      return <UnifiedCanvasParameters />;
    }

    return null;
  }, [activeTabName, model]);

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
