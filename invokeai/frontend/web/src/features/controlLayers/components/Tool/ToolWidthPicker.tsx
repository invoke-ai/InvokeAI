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
import useResizeObserver from '@react-hook/resize-observer';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { clamp } from 'es-toolkit/compat';
import {
  selectCanvasSettingsSlice,
  settingsBrushWidthChanged,
  settingsEraserWidthChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import type { KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PiCaretDownBold } from 'react-icons/pi';

import { useToolIsSelected } from './hooks';

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

type ToolWidthPickerComponent = 'dropDown' | 'slider';
const SLIDER_PICKER_WIDTH = 280;
const DROPDOWN_PICKER_WIDTH = 76;
const MIN_TOOLBAR_SPACE = 50;

interface ToolWidthPickerComponentProps {
  localValue: number;
  onChangeSlider: (value: number) => void;
  onChangeInput: (value: number) => void;
  onBlur: () => void;
  onKeyDown: (value: KeyboardEvent<HTMLInputElement>) => void;
  onComponentWidthChange: (value: number) => void;
}

const DropDownToolWidthPickerComponent = memo(
  ({
    localValue,
    onChangeSlider,
    onChangeInput,
    onKeyDown,
    onBlur,
    onComponentWidthChange,
  }: ToolWidthPickerComponentProps) => {
    const ref = useRef(null);

    useResizeObserver(ref, (entry) => onComponentWidthChange(entry.contentRect.width));

    const onChangeNumberInput = useCallback(
      (valueAsString: string, valueAsNumber: number) => {
        onChangeInput(valueAsNumber);
      },
      [onChangeInput]
    );

    return (
      <Popover>
        <FormControl ref={ref} w="min-content" gap={2}>
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
              w={DROPDOWN_PICKER_WIDTH}
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
DropDownToolWidthPickerComponent.displayName = 'DropDownToolWidthPickerComponent';

const SliderToolWidthPickerComponent = memo(
  ({
    localValue,
    onChangeSlider,
    onChangeInput,
    onKeyDown,
    onBlur,
    onComponentWidthChange,
  }: ToolWidthPickerComponentProps) => {
    const ref = useRef(null);

    useResizeObserver(ref, (entry) => onComponentWidthChange(entry.contentRect.width));

    return (
      <Flex ref={ref} w={SLIDER_PICKER_WIDTH} h={8} gap={4}>
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
          w={28}
          variant="outline"
          min={1}
          max={600}
          value={localValue}
          onChange={onChangeInput}
          onBlur={onBlur}
          onKeyDown={onKeyDown}
          format={formatPx}
          defaultValue={50}
        />
      </Flex>
    );
  }
);
SliderToolWidthPickerComponent.displayName = 'SliderToolWidthPickerComponent';

const components = {
  dropDown: DropDownToolWidthPickerComponent,
  slider: SliderToolWidthPickerComponent,
} as const;

const calculateToolWidthPicker = (current: ToolWidthPickerComponent | undefined, toolBarAvailableSpace: number) => {
  switch (current) {
    case 'dropDown':
      if (toolBarAvailableSpace + DROPDOWN_PICKER_WIDTH > SLIDER_PICKER_WIDTH + MIN_TOOLBAR_SPACE) {
        return 'slider';
      }
      break;
    case 'slider':
      if (toolBarAvailableSpace < MIN_TOOLBAR_SPACE) {
        return 'dropDown';
      }
      break;
    default:
      return toolBarAvailableSpace > SLIDER_PICKER_WIDTH + MIN_TOOLBAR_SPACE ? 'slider' : 'dropDown';
  }

  return current;
};

const useToolWidthPicker = (toolBarAvailableSpace: number) => {
  const prevToolWidthPicker = useRef<ToolWidthPickerComponent>();
  const toolWidthPicker = useMemo<ToolWidthPickerComponent>(() => {
    const next = calculateToolWidthPicker(prevToolWidthPicker.current, toolBarAvailableSpace);

    prevToolWidthPicker.current = next;

    return next;
  }, [toolBarAvailableSpace]);

  return toolWidthPicker;
};

const selectBrushWidth = createSelector(selectCanvasSettingsSlice, (settings) => settings.brushWidth);
const selectEraserWidth = createSelector(selectCanvasSettingsSlice, (settings) => settings.eraserWidth);

interface ToolWidthPickerProps {
  toolBarAvailableSpace: number;
  onComponentWidthChange: (value: number) => void;
}

export const ToolWidthPicker = memo(({ toolBarAvailableSpace, onComponentWidthChange }: ToolWidthPickerProps) => {
  const dispatch = useAppDispatch();
  const isBrushSelected = useToolIsSelected('brush');
  const isEraserSelected = useToolIsSelected('eraser');
  const isToolSelected = useMemo(() => {
    return isBrushSelected || isEraserSelected;
  }, [isBrushSelected, isEraserSelected]);
  const brushWidth = useAppSelector(selectBrushWidth);
  const eraserWidth = useAppSelector(selectEraserWidth);
  const width = useMemo(() => {
    if (isBrushSelected) {
      return brushWidth;
    }
    if (isEraserSelected) {
      return eraserWidth;
    }
    return 0;
  }, [isBrushSelected, isEraserSelected, brushWidth, eraserWidth]);
  const [localValue, setLocalValue] = useState(width);
  const toolWidthPicker = useToolWidthPicker(toolBarAvailableSpace);

  const onValueChange = useCallback(
    (value: number) => {
      if (isBrushSelected) {
        dispatch(settingsBrushWidthChanged(value));
      } else if (isEraserSelected) {
        dispatch(settingsEraserWidthChanged(value));
      }
    },
    [isBrushSelected, isEraserSelected, dispatch]
  );

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
    options: { enabled: isToolSelected },
    dependencies: [decrement, isToolSelected],
  });
  useRegisteredHotkeys({
    id: 'incrementToolWidth',
    category: 'canvas',
    callback: increment,
    options: { enabled: isToolSelected },
    dependencies: [increment, isToolSelected],
  });

  const Component = components[toolWidthPicker];

  return (
    <Flex px={4}>
      <Component
        localValue={localValue}
        onChangeSlider={onChangeSlider}
        onChangeInput={onChangeInput}
        onBlur={onBlur}
        onKeyDown={onKeyDown}
        onComponentWidthChange={onComponentWidthChange}
      />
    </Flex>
  );
});

ToolWidthPicker.displayName = 'ToolWidthPicker';
