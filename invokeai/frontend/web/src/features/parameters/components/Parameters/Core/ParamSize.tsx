import { Flex, Spacer, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import {
  setAspectRatio,
  setShouldLockAspectRatio,
  toggleSize,
} from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaLock } from 'react-icons/fa';
import { MdOutlineSwapVert } from 'react-icons/md';
import { activeTabNameSelector } from '../../../../ui/store/uiSelectors';
import ParamAspectRatio from './ParamAspectRatio';
import ParamHeight from './ParamHeight';
import ParamWidth from './ParamWidth';

const sizeOptsSelector = createSelector(
  [generationSelector, activeTabNameSelector],
  (generation, activeTabName) => {
    const { shouldFitToWidthHeight, shouldLockAspectRatio, width, height } =
      generation;

    return {
      activeTabName,
      shouldFitToWidthHeight,
      shouldLockAspectRatio,
      width,
      height,
    };
  }
);

export default function ParamSize() {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const {
    activeTabName,
    shouldFitToWidthHeight,
    shouldLockAspectRatio,
    width,
    height,
  } = useAppSelector(sizeOptsSelector);

  const handleLockRatio = useCallback(() => {
    dispatch(setAspectRatio(width / height));
    dispatch(setShouldLockAspectRatio(!shouldLockAspectRatio));
  }, [shouldLockAspectRatio, width, height, dispatch]);

  return (
    <Flex
      sx={{
        gap: 2,
        p: 4,
        borderRadius: 4,
        flexDirection: 'column',
        w: 'full',
        bg: 'base.150',
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
          isDisabled={
            activeTabName === 'img2img' ? !shouldFitToWidthHeight : false
          }
          onClick={() => dispatch(toggleSize())}
        />
        <IAIIconButton
          tooltip={t('ui.lockRatio')}
          aria-label={t('ui.lockRatio')}
          size="sm"
          icon={<FaLock />}
          isChecked={shouldLockAspectRatio}
          isDisabled={
            activeTabName === 'img2img' ? !shouldFitToWidthHeight : false
          }
          onClick={handleLockRatio}
        />
      </Flex>
      <Flex gap={2} alignItems="center">
        <Flex gap={2} flexDirection="column" width="full">
          <ParamWidth
            isDisabled={
              activeTabName === 'img2img' ? !shouldFitToWidthHeight : false
            }
          />
          <ParamHeight
            isDisabled={
              activeTabName === 'img2img' ? !shouldFitToWidthHeight : false
            }
          />
        </Flex>
      </Flex>
    </Flex>
  );
}
