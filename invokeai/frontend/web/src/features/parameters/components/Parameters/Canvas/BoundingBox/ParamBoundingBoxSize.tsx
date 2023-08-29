import { Flex, Spacer, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
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
} from '../../Core/ParamAspectRatio';
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
      <Flex alignItems="center" gap={2}>
        <Text
          sx={{
            fontSize: 'sm',
            width: 'full',
            color: 'base.700',
            _dark: {
              color: 'base.300',
            },
          }}
        >
          {t('parameters.aspectRatio')}
        </Text>
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
      </Flex>
      <ParamBoundingBoxWidth />
      <ParamBoundingBoxHeight />
    </Flex>
  );
}
