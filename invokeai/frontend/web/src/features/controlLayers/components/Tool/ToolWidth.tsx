import {
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
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
import { useAppSelector } from 'app/store/storeHooks';
import { clamp } from 'es-toolkit/compat';
import { selectToolWidthSelector } from 'features/controlLayers/store/canvasSettingsSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { PiCaretDownBold } from 'react-icons/pi';

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

interface ToolWidthSelectorProps {
  localValue: number;
  onChangeSlider: (value: number) => void;
  onChangeInput: (value: number) => void;
  onBlur: () => void;
  onKeyDown: (value: KeyboardEvent<HTMLInputElement>) => void;
}

const DropDownToolWidthSelector = memo(
  ({ localValue, onChangeSlider, onChangeInput, onKeyDown, onBlur }: ToolWidthSelectorProps) => {
    const onChangeNumberInput = useCallback(
      (valueAsString: string, valueAsNumber: number) => {
        onChangeInput(valueAsNumber);
      },
      [onChangeInput]
    );

    return (
      <Popover>
        <FormControl w="min-content" gap={2}>
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
  }
);
DropDownToolWidthSelector.displayName = 'DropDownToolWidthSelector';

const SliderToolWidthSelector = memo(
  ({ localValue, onChangeSlider, onChangeInput, onKeyDown, onBlur }: ToolWidthSelectorProps) => {
    return (
      <Flex w="full" gap={4} alignItems="center" px={4}>
        <CompositeSlider
          w={200}
          min={0}
          max={100}
          value={mapRawValueToSliderValue(localValue)}
          onChange={onChangeSlider}
          defaultValue={sliderDefaultValue}
          marks={marks}
          formatValue={formatSliderValue}
          alwaysShowMarks
        />
        <CompositeNumberInput
          min={1}
          max={600}
          value={localValue}
          onChange={onChangeInput}
          onBlur={onBlur}
          onKeyDown={onKeyDown}
          w={24}
          format={formatPx}
          defaultValue={50}
        />
      </Flex>
    );
  }
);
SliderToolWidthSelector.displayName = 'SliderToolWidthSelector';

const selectorComponents = {
  dropDown: DropDownToolWidthSelector,
  slider: SliderToolWidthSelector,
} as const;

interface ToolWidthProps {
  isSelected: boolean;
  width: number;
  onValueChange: (value: number) => void;
}

export const ToolWidth = memo(({ isSelected, width, onValueChange }: ToolWidthProps) => {
  const toolWidthSelector = useAppSelector(selectToolWidthSelector);
  const [localValue, setLocalValue] = useState(width);

  const onChange = useCallback(
    (value: number) => {
      onValueChange(clamp(Math.round(value), 1, 600));
    },
    [onValueChange]
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

  const onChangeInput = useCallback((value: number) => {
    setLocalValue(value);
  }, []);

  const onBlur = useCallback(() => {
    if (isNaN(Number(localValue))) {
      onChange(50);
      setLocalValue(50);
    } else {
      onChange(localValue);
    }
  }, [localValue, onChange]);

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
    options: { enabled: isSelected },
    dependencies: [decrement, isSelected],
  });
  useRegisteredHotkeys({
    id: 'incrementToolWidth',
    category: 'canvas',
    callback: increment,
    options: { enabled: isSelected },
    dependencies: [increment, isSelected],
  });

  const Component = selectorComponents[toolWidthSelector];

  return (
    <Component
      localValue={localValue}
      onChangeSlider={onChangeSlider}
      onChangeInput={onChangeInput}
      onBlur={onBlur}
      onKeyDown={onKeyDown}
    />
  );
});

ToolWidth.displayName = 'ToolWidth';
