import {
  CompositeSlider,
  FormControl,
  FormLabel,
  IconButton,
  NumberInput,
  NumberInputField,
  Popover,
  PopoverAnchor,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { selectCanvasSettingsSlice, settingsBrushWidthChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { clamp } from 'lodash-es';
import type { KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

const selectBrushWidth = createSelector(selectCanvasSettingsSlice, (settings) => settings.brushWidth);
const formatPx = (v: number | string) => `${v} px`;

function mapSliderValueToRawValue(value: number) {
  if (value <= 40) {
    // 0 to 40 on the slider -> 1px to 50px
    return 1 + (49 * value) / 40;
  } else if (value <= 70) {
    // 40 to 70 on the slider -> 50px to 200px
    return 50 + (150 * (value - 40)) / 30;
  } else {
    // 70 to 100 on the slider -> 200px to 600px
    return 200 + (400 * (value - 70)) / 30;
  }
}

function mapRawValueToSliderValue(value: number) {
  if (value <= 50) {
    // 1px to 50px -> 0 to 40 on the slider
    return ((value - 1) * 40) / 49;
  } else if (value <= 200) {
    // 50px to 200px -> 40 to 70 on the slider
    return 40 + ((value - 50) * 30) / 150;
  } else {
    // 200px to 600px -> 70 to 100 on the slider
    return 70 + ((value - 200) * 30) / 400;
  }
}

function formatSliderValue(value: number) {
  return `${String(mapSliderValueToRawValue(value))} px`;
}

const marks = [
  mapRawValueToSliderValue(1),
  mapRawValueToSliderValue(50),
  mapRawValueToSliderValue(200),
  mapRawValueToSliderValue(600),
];

const sliderDefaultValue = mapRawValueToSliderValue(50);

export const ToolBrushWidth = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const imageViewer = useImageViewer();
  const isSelected = useToolIsSelected('brush');
  const width = useAppSelector(selectBrushWidth);
  const [localValue, setLocalValue] = useState(width);
  const onChange = useCallback(
    (v: number) => {
      dispatch(settingsBrushWidthChanged(clamp(Math.round(v), 1, 600)));
    },
    [dispatch]
  );

  const increment = useCallback(() => {
    let newWidth = Math.round(width * 1.15);
    if (newWidth === width) {
      newWidth += 1;
    }
    onChange(newWidth);
  }, [onChange, width]);

  const decrement = useCallback(() => {
    let newWidth = Math.round(width * 0.85);
    if (newWidth === width) {
      newWidth -= 1;
    }
    onChange(newWidth);
  }, [onChange, width]);

  const onChangeSlider = useCallback(
    (value: number) => {
      onChange(mapSliderValueToRawValue(value));
    },
    [onChange]
  );

  const onBlur = useCallback(() => {
    if (isNaN(Number(localValue))) {
      onChange(50);
      setLocalValue(50);
    } else {
      onChange(localValue);
    }
  }, [localValue, onChange]);

  const onChangeNumberInput = useCallback((valueAsString: string, valueAsNumber: number) => {
    setLocalValue(valueAsNumber);
  }, []);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        onBlur();
      }
    },
    [onBlur]
  );

  useEffect(() => {
    setLocalValue(width);
  }, [width]);

  useRegisteredHotkeys({
    id: 'decrementToolWidth',
    category: 'canvas',
    callback: decrement,
    options: { enabled: isSelected && !imageViewer.isOpen },
    dependencies: [decrement, isSelected, imageViewer.isOpen],
  });
  useRegisteredHotkeys({
    id: 'incrementToolWidth',
    category: 'canvas',
    callback: increment,
    options: { enabled: isSelected && !imageViewer.isOpen },
    dependencies: [increment, isSelected, imageViewer.isOpen],
  });

  return (
    <Popover>
      <FormControl w="min-content" gap={2}>
        <FormLabel m={0}>{t('controlLayers.width')}</FormLabel>
        <PopoverAnchor>
          <NumberInput
            variant="outline"
            display="flex"
            alignItems="center"
            min={1}
            max={600}
            value={localValue}
            onChange={onChangeNumberInput}
            onBlur={onBlur}
            w="76px"
            format={formatPx}
            defaultValue={50}
            onKeyDown={onKeyDown}
            clampValueOnBlur={false}
          >
            <NumberInputField _focusVisible={{ zIndex: 0 }} title="" paddingInlineEnd={7} />
            <PopoverTrigger>
              <IconButton
                aria-label="open-slider"
                icon={<PiCaretDownBold />}
                size="sm"
                variant="link"
                position="absolute"
                insetInlineEnd={0}
                h="full"
              />
            </PopoverTrigger>
          </NumberInput>
        </PopoverAnchor>
      </FormControl>
      <PopoverContent w={200} pt={0} pb={2} px={4}>
        <PopoverArrow />
        <PopoverBody>
          <CompositeSlider
            min={0}
            max={100}
            value={mapRawValueToSliderValue(localValue)}
            onChange={onChangeSlider}
            defaultValue={sliderDefaultValue}
            marks={marks}
            formatValue={formatSliderValue}
            alwaysShowMarks
          />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

ToolBrushWidth.displayName = 'ToolBrushWidth';
