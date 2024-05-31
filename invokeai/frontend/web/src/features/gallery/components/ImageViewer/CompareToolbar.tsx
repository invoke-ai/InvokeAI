import { Button, ButtonGroup, Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  comparedImagesSwapped,
  comparisonModeChanged,
  imageToCompareChanged,
  sliderFitChanged,
} from 'features/gallery/store/gallerySlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutBold, PiSwapBold, PiXBold } from 'react-icons/pi';

export const CompareToolbar = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const comparisonMode = useAppSelector((s) => s.gallery.comparisonMode);
  const sliderFit = useAppSelector((s) => s.gallery.sliderFit);
  const setComparisonModeSlider = useCallback(() => {
    dispatch(comparisonModeChanged('slider'));
  }, [dispatch]);
  const setComparisonModeSideBySide = useCallback(() => {
    dispatch(comparisonModeChanged('side-by-side'));
  }, [dispatch]);
  const swapImages = useCallback(() => {
    dispatch(comparedImagesSwapped());
  }, [dispatch]);
  const toggleSliderFit = useCallback(() => {
    dispatch(sliderFitChanged(sliderFit === 'contain' ? 'fill' : 'contain'));
  }, [dispatch, sliderFit]);
  const exitCompare = useCallback(() => {
    dispatch(imageToCompareChanged(null));
  }, [dispatch]);

  return (
    <Flex w="full" gap={2}>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineEnd="auto">
          <IconButton
            icon={<PiSwapBold />}
            aria-label={t('gallery.swapImages')}
            tooltip={t('gallery.swapImages')}
            onClick={swapImages}
          />
          {comparisonMode === 'slider' && (
            <IconButton
              aria-label={t('gallery.stretchToFit')}
              tooltip={t('gallery.stretchToFit')}
              isDisabled={comparisonMode !== 'slider'}
              onClick={toggleSliderFit}
              colorScheme={sliderFit === 'fill' ? 'invokeBlue' : 'base'}
              variant="outline"
              icon={<PiArrowsOutBold />}
            />
          )}
        </Flex>
      </Flex>
      <Flex flex={1} gap={4} justifyContent="center">
        <ButtonGroup variant="outline">
          <Button
            flexShrink={0}
            onClick={setComparisonModeSlider}
            colorScheme={comparisonMode === 'slider' ? 'invokeBlue' : 'base'}
          >
            {t('gallery.slider')}
          </Button>
          <Button
            flexShrink={0}
            onClick={setComparisonModeSideBySide}
            colorScheme={comparisonMode === 'side-by-side' ? 'invokeBlue' : 'base'}
          >
            {t('gallery.sideBySide')}
          </Button>
        </ButtonGroup>
      </Flex>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineStart="auto">
          <IconButton
            icon={<PiXBold />}
            aria-label={t('gallery.exitCompare')}
            tooltip={t('gallery.exitCompare')}
            onClick={exitCompare}
          />
        </Flex>
      </Flex>
    </Flex>
  );
});

CompareToolbar.displayName = 'CompareToolbar';
