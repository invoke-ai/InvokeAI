import {
  Button,
  ButtonGroup,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Switch,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  comparedImagesSwapped,
  comparisonModeChanged,
  imageToCompareChanged,
  sliderFitChanged,
} from 'features/gallery/store/gallerySlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearBold } from 'react-icons/pi';

export const ImageComparisonToolbarButtons = memo(() => {
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
  const onSliderFitChanged = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(sliderFitChanged(e.target.checked ? 'fill' : 'contain'));
    },
    [dispatch]
  );
  const exitCompare = useCallback(() => {
    dispatch(imageToCompareChanged(null));
  }, [dispatch]);

  return (
    <>
      <Popover isLazy>
        <PopoverTrigger>
          <IconButton
            aria-label={t('gallery.compareOptions')}
            tooltip={t('gallery.compareOptions')}
            icon={<PiGearBold />}
          />
        </PopoverTrigger>
        <PopoverContent>
          <PopoverBody>
            <Flex direction="column" gap={2}>
              <ButtonGroup variant="outline" size="sm" w="full">
                <Button
                  flex={1}
                  onClick={setComparisonModeSlider}
                  colorScheme={comparisonMode === 'slider' ? 'invokeBlue' : 'base'}
                >
                  {t('gallery.slider')}
                </Button>
                <Button
                  flex={1}
                  onClick={setComparisonModeSideBySide}
                  colorScheme={comparisonMode === 'side-by-side' ? 'invokeBlue' : 'base'}
                >
                  {t('gallery.sideBySide')}
                </Button>
              </ButtonGroup>
              <FormControl isDisabled={comparisonMode !== 'slider'}>
                <FormLabel>{t('gallery.sliderFitLabel')}</FormLabel>
                <Switch isChecked={sliderFit === 'fill'} onChange={onSliderFitChanged} />
              </FormControl>
            </Flex>
          </PopoverBody>
        </PopoverContent>
      </Popover>

      <Button onClick={swapImages}>{t('gallery.swapImages')}</Button>
      <Button onClick={exitCompare}>{t('gallery.exitCompare')}</Button>
    </>
  );
});

ImageComparisonToolbarButtons.displayName = 'ImageComparisonToolbarButtons';
