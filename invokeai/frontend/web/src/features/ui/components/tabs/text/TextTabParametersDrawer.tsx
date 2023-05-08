import { isEqual } from 'lodash-es';
import ResizableDrawer from '../../common/ResizableDrawer/ResizableDrawer';
import TextTabParameters from './TextTabParameters';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector, uiSelector } from '../../../store/uiSelectors';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setShouldShowParametersPanel } from '../../../store/uiSlice';
import { memo } from 'react';
import { Flex } from '@chakra-ui/react';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import { Panel, PanelGroup } from 'react-resizable-panels';

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
      activeTabName,
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

const TextTabParametersDrawer = () => {
  const dispatch = useAppDispatch();
  const { shouldPinParametersPanel, shouldShowParametersPanel } =
    useAppSelector(selector);

  const handleClosePanel = () => {
    dispatch(setShouldShowParametersPanel(false));
  };

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
        <TextTabParameters />
      </Flex>
    </ResizableDrawer>
  );
};

export default memo(TextTabParametersDrawer);
