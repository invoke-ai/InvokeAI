import { ChakraProps, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import InvokeButton from 'features/parameters/components/ProcessButtons/InvokeButton';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { setShouldShowParametersPanel } from 'features/ui/store/uiSlice';
import { isEqual } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { FaSlidersH } from 'react-icons/fa';

const floatingButtonStyles: ChakraProps['sx'] = {
  borderStartStartRadius: 0,
  borderEndStartRadius: 0,
};

export const floatingParametersPanelButtonSelector = createSelector(
  [uiSelector, activeTabNameSelector],
  (ui, activeTabName) => {
    const {
      shouldPinParametersPanel,
      shouldUseCanvasBetaLayout,
      shouldShowParametersPanel,
    } = ui;

    const canvasBetaLayoutCheck =
      shouldUseCanvasBetaLayout && activeTabName === 'unifiedCanvas';

    const shouldShowProcessButtons =
      !canvasBetaLayoutCheck &&
      (!shouldPinParametersPanel || !shouldShowParametersPanel);

    const shouldShowParametersPanelButton =
      !canvasBetaLayoutCheck &&
      !shouldShowParametersPanel &&
      ['txt2img', 'img2img', 'unifiedCanvas'].includes(activeTabName);

    return {
      shouldPinParametersPanel,
      shouldShowParametersPanelButton,
      shouldShowProcessButtons,
    };
  },
  { memoizeOptions: { resultEqualityCheck: isEqual } }
);

const FloatingParametersPanelButtons = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const {
    shouldShowProcessButtons,
    shouldShowParametersPanelButton,
    shouldPinParametersPanel,
  } = useAppSelector(floatingParametersPanelButtonSelector);

  const handleShowOptionsPanel = () => {
    dispatch(setShouldShowParametersPanel(true));
    shouldPinParametersPanel && dispatch(requestCanvasRescale());
  };

  if (!shouldShowParametersPanelButton) {
    return null;
  }

  return (
    <Flex
      pos="absolute"
      transform="translate(0, -50%)"
      minW={8}
      top="50%"
      insetInlineStart="4.5rem"
      direction="column"
      gap={2}
    >
      <IAIIconButton
        tooltip="Show Options Panel (O)"
        tooltipProps={{ placement: 'top' }}
        aria-label={t('accessibility.showOptionsPanel')}
        onClick={handleShowOptionsPanel}
        sx={floatingButtonStyles}
      >
        <FaSlidersH />
      </IAIIconButton>
      {shouldShowProcessButtons && (
        <>
          <InvokeButton iconButton sx={floatingButtonStyles} />
          <CancelButton sx={floatingButtonStyles} />
        </>
      )}
    </Flex>
  );
};

export default memo(FloatingParametersPanelButtons);
