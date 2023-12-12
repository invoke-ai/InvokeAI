import { Flex, FormControl, FormLabel, Spacer } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import {
  setAspectRatio,
  setShouldLockAspectRatio,
  toggleSize,
} from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaLock } from 'react-icons/fa';
import { MdOutlineSwapVert } from 'react-icons/md';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import ParamAspectRatio, { mappedAspectRatios } from './ParamAspectRatio';
import ParamHeight from './ParamHeight';
import ParamWidth from './ParamWidth';

const sizeOptsSelector = createMemoizedSelector(
  [stateSelector, activeTabNameSelector],
  ({ generation }, activeTabName) => {
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
    if (shouldLockAspectRatio) {
      dispatch(setShouldLockAspectRatio(false));
      if (!mappedAspectRatios.includes(width / height)) {
        dispatch(setAspectRatio(null));
      } else {
        dispatch(setAspectRatio(width / height));
      }
    } else {
      dispatch(setShouldLockAspectRatio(true));
      dispatch(setAspectRatio(width / height));
    }
  }, [shouldLockAspectRatio, width, height, dispatch]);

  const handleToggleSize = useCallback(() => {
    dispatch(toggleSize());
    dispatch(setAspectRatio(null));
    if (shouldLockAspectRatio) {
      dispatch(setAspectRatio(height / width));
    }
  }, [dispatch, shouldLockAspectRatio, width, height]);

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
            isDisabled={
              activeTabName === 'img2img' ? !shouldFitToWidthHeight : false
            }
            onClick={handleToggleSize}
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
        </FormControl>
      </IAIInformationalPopover>
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
