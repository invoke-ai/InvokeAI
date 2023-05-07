import { isEqual } from 'lodash-es';
import ResizableDrawer from './common/ResizableDrawer/ResizableDrawer';
import GenerateParameters from './tabs/Create/GenerateParameters';
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

const selector = createSelector(
  [uiSelector, activeTabNameSelector, lightboxSelector],
  (ui, activeTabName, lightbox) => {
    const { shouldPinParametersPanel, shouldShowParametersPanel } = ui;
    const { isLightboxOpen } = lightbox;

    return {
      shouldPinParametersPanel,
      shouldShowParametersPanel,
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
  const { shouldPinParametersPanel, shouldShowParametersPanel } =
    useAppSelector(selector);

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
      <Flex
        flexDir="column"
        position="relative"
        h={{ base: 600, xl: 'full' }}
        w={{ sm: 'full', lg: '100vw', xl: 'full' }}
        paddingRight={{ base: 8, xl: 0 }}
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
        <GenerateParameters />
      </Flex>
    </ResizableDrawer>
  );
};

export default memo(CreateParametersPanel);
