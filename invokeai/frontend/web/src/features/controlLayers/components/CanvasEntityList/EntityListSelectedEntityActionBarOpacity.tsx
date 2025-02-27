import {
  $shift,
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
import { snapToNearest } from 'features/controlLayers/konva/util';
import { entityOpacityChanged } from 'features/controlLayers/store/canvasSlice';
import {
  selectCanvasSlice,
  selectEntity,
  selectSelectedEntityIdentifier,
} from 'features/controlLayers/store/selectors';
import { isRenderableEntity } from 'features/controlLayers/store/types';
import { clamp, round } from 'lodash-es';
import type { KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

function formatPct(v: number | string) {
  if (isNaN(Number(v))) {
    return '';
  }

  return `${round(Number(v), 2).toLocaleString()}%`;
}

function mapSliderValueToRawValue(value: number) {
  return value / 100;
}

function mapRawValueToSliderValue(opacity: number) {
  return opacity * 100;
}

function formatSliderValue(value: number) {
  return String(value);
}

const marks = [
  mapRawValueToSliderValue(0),
  mapRawValueToSliderValue(0.25),
  mapRawValueToSliderValue(0.5),
  mapRawValueToSliderValue(0.75),
  mapRawValueToSliderValue(1),
];

const sliderDefaultValue = mapRawValueToSliderValue(1);

const snapCandidates = marks.slice(1, marks.length - 1);

const selectOpacity = createSelector(selectCanvasSlice, (canvas) => {
  const selectedEntityIdentifier = canvas.selectedEntityIdentifier;
  if (!selectedEntityIdentifier) {
    return 1; // fallback to 100% opacity
  }
  const selectedEntity = selectEntity(canvas, selectedEntityIdentifier);
  if (!selectedEntity) {
    return 1; // fallback to 100% opacity
  }
  if (!isRenderableEntity(selectedEntity)) {
    return 1; // fallback to 100% opacity
  }
  // Opacity is a float from 0-1, but we want to display it as a percentage
  return selectedEntity.opacity;
});

export const EntityListSelectedEntityActionBarOpacity = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const opacity = useAppSelector(selectOpacity);

  const [localOpacity, setLocalOpacity] = useState(opacity * 100);

  const onChangeSlider = useCallback(
    (opacity: number) => {
      if (!selectedEntityIdentifier) {
        return;
      }
      let snappedOpacity = opacity;
      // Do not snap if shift key is held
      if (!$shift.get()) {
        snappedOpacity = snapToNearest(opacity, snapCandidates, 2);
      }
      const mappedOpacity = mapSliderValueToRawValue(snappedOpacity);

      dispatch(entityOpacityChanged({ entityIdentifier: selectedEntityIdentifier, opacity: mappedOpacity }));
    },
    [dispatch, selectedEntityIdentifier]
  );

  const onBlur = useCallback(() => {
    if (!selectedEntityIdentifier) {
      return;
    }
    if (isNaN(Number(localOpacity))) {
      setLocalOpacity(100);
      return;
    }
    dispatch(
      entityOpacityChanged({ entityIdentifier: selectedEntityIdentifier, opacity: clamp(localOpacity / 100, 0, 1) })
    );
  }, [dispatch, localOpacity, selectedEntityIdentifier]);

  const onChangeNumberInput = useCallback((valueAsString: string, valueAsNumber: number) => {
    setLocalOpacity(valueAsNumber);
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
    setLocalOpacity((opacity ?? 1) * 100);
  }, [opacity]);

  return (
    <Popover>
      <FormControl
        w="min-content"
        gap={2}
        isDisabled={selectedEntityIdentifier === null || selectedEntityIdentifier.type === 'reference_image'}
      >
        <FormLabel m={0}>{t('controlLayers.opacity')}</FormLabel>
        <PopoverAnchor>
          <NumberInput
            display="flex"
            alignItems="center"
            min={0}
            max={100}
            step={1}
            value={localOpacity}
            onChange={onChangeNumberInput}
            onBlur={onBlur}
            w="76px"
            format={formatPct}
            defaultValue={1}
            onKeyDown={onKeyDown}
            clampValueOnBlur={false}
            variant="outline"
          >
            <NumberInputField paddingInlineEnd={7} _focusVisible={{ zIndex: 0 }} title="" />
            <PopoverTrigger>
              <IconButton
                aria-label="open-slider"
                icon={<PiCaretDownBold />}
                size="sm"
                variant="link"
                position="absolute"
                insetInlineEnd={0}
                h="full"
                isDisabled={selectedEntityIdentifier === null || selectedEntityIdentifier.type === 'reference_image'}
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
            value={localOpacity}
            onChange={onChangeSlider}
            defaultValue={sliderDefaultValue}
            marks={marks}
            formatValue={formatSliderValue}
            alwaysShowMarks
            isDisabled={selectedEntityIdentifier === null || selectedEntityIdentifier.type === 'reference_image'}
          />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

EntityListSelectedEntityActionBarOpacity.displayName = 'EntityListSelectedEntityActionBarOpacity';
