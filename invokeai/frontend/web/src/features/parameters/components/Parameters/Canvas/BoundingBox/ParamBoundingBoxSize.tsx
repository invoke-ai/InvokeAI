import { Flex, FormControl, FormLabel, Spacer } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { flipBoundingBoxAxes } from 'features/canvas/store/canvasSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import {
  setAspectRatio,
  setShouldLockAspectRatio,
} from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaLock } from 'react-icons/fa';
import { MdOutlineSwapVert } from 'react-icons/md';
import ParamAspectRatio, {
  mappedAspectRatios,
} from 'features/parameters/components/Parameters/Core/ParamAspectRatio';
import ParamBoundingBoxHeight from './ParamBoundingBoxHeight';
import ParamBoundingBoxWidth from './ParamBoundingBoxWidth';

const sizeOptsSelector = createSelector(
  [generationSelector, canvasSelector],
  (generation, canvas) => {
    const { shouldFitToWidthHeight, shouldLockAspectRatio } = generation;
    const { boundingBoxDimensions } = canvas;

    return {
      shouldFitToWidthHeight,
      shouldLockAspectRatio,
      boundingBoxDimensions,
    };
  }
);

export default function ParamBoundingBoxSize() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const { shouldLockAspectRatio, boundingBoxDimensions } =
    useAppSelector(sizeOptsSelector);

  const handleLockRatio = useCallback(() => {
    if (shouldLockAspectRatio) {
      dispatch(setShouldLockAspectRatio(false));
      if (
        !mappedAspectRatios.includes(
          boundingBoxDimensions.width / boundingBoxDimensions.height
        )
      ) {
        dispatch(setAspectRatio(null));
      } else {
        dispatch(
          setAspectRatio(
            boundingBoxDimensions.width / boundingBoxDimensions.height
          )
        );
      }
    } else {
      dispatch(setShouldLockAspectRatio(true));
      dispatch(
        setAspectRatio(
          boundingBoxDimensions.width / boundingBoxDimensions.height
        )
      );
    }
  }, [shouldLockAspectRatio, boundingBoxDimensions, dispatch]);

  const handleToggleSize = useCallback(() => {
    dispatch(flipBoundingBoxAxes());
    dispatch(setAspectRatio(null));
    if (shouldLockAspectRatio) {
      dispatch(
        setAspectRatio(
          boundingBoxDimensions.height / boundingBoxDimensions.width
        )
      );
    }
  }, [dispatch, shouldLockAspectRatio, boundingBoxDimensions]);

  return (
    <Flex
      sx={{
        gap: 2,
        p: 4,
        borderRadius: 4,
        flexDirection: 'column',
        w: 'full',
        bg: 'base.100',
        _dark: {
          bg: 'base.750',
        },
      }}
    >
      <IAIInformationalPopover feature="paramRatio">
        <FormControl as={Flex} flexDir="row" alignItems="center" gap={2}>
          <FormLabel>{t('parameters.aspectRatio')}</FormLabel>
          <Spacer />
          <ParamAspectRatio />
          <IAIIconButton
            tooltip={t('ui.swapSizes')}
            aria-label={t('ui.swapSizes')}
            size="sm"
            icon={<MdOutlineSwapVert />}
            fontSize={20}
            onClick={handleToggleSize}
          />
          <IAIIconButton
            tooltip={t('ui.lockRatio')}
            aria-label={t('ui.lockRatio')}
            size="sm"
            icon={<FaLock />}
            isChecked={shouldLockAspectRatio}
            onClick={handleLockRatio}
          />
        </FormControl>
      </IAIInformationalPopover>
      <ParamBoundingBoxWidth />
      <ParamBoundingBoxHeight />
    </Flex>
  );
}
