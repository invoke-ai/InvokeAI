import {
  Checkbox,
  CompositeSlider,
  FormControl,
  FormLabel,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Popover,
  PopoverAnchor,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ToolWidthPicker } from 'features/controlLayers/components/Tool/ToolWidthPicker';
import {
  selectTraceTaper,
  selectTraceTaperEnds,
  settingsTraceTaperChanged,
  settingsTraceTaperEndsToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { ChangeEventHandler, FocusEvent, KeyboardEvent } from 'react';
import { Fragment, memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { MdKeyboardArrowDown, MdKeyboardArrowUp } from 'react-icons/md';

const formatPercent = (value: number | string) => (Number.isNaN(Number(value)) ? '' : `${value}%`);
const taperMarks = [1, 100, 250, 500];
const selectInputValue = (event: FocusEvent<HTMLInputElement>) => {
  event.currentTarget.select();
};

export const VectorLayerTraceWidth = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const traceTaperEnds = useAppSelector(selectTraceTaperEnds);
  const traceTaper = useAppSelector(selectTraceTaper);
  const [localTaper, setLocalTaper] = useState(traceTaper);
  const label = t('controlLayers.vectorEdit.traceWidth');
  const onTaperEndsChange = useCallback<ChangeEventHandler<HTMLInputElement>>(() => {
    dispatch(settingsTraceTaperEndsToggled());
  }, [dispatch]);
  const commitTaper = useCallback(() => {
    const nextTaper = Number.isFinite(localTaper) ? localTaper : traceTaper;
    dispatch(settingsTraceTaperChanged(nextTaper));
  }, [dispatch, localTaper, traceTaper]);
  const onTaperChange = useCallback((_valueAsString: string, valueAsNumber: number) => {
    setLocalTaper(valueAsNumber);
  }, []);
  const onTaperSliderChange = useCallback(
    (value: number) => {
      setLocalTaper(value);
      dispatch(settingsTraceTaperChanged(value));
    },
    [dispatch]
  );
  const onTaperKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') {
        commitTaper();
      }
    },
    [commitTaper]
  );

  useEffect(() => {
    setLocalTaper(traceTaper);
  }, [traceTaper]);

  return (
    <Fragment>
      <FormControl w="min-content" flexShrink={0} gap={2}>
        <FormLabel m={0} mt={1} whiteSpace="nowrap">
          {label}
        </FormLabel>
        <ToolWidthPicker mode="trace" ariaLabel={label} />
      </FormControl>
      <FormControl w="min-content" flexShrink={0} gap={2}>
        <FormLabel m={0} whiteSpace="nowrap">
          {t('controlLayers.vectorEdit.taperEnds')}
        </FormLabel>
        <Checkbox isChecked={traceTaperEnds} onChange={onTaperEndsChange} />
      </FormControl>
      <Popover>
        <FormControl w="min-content" flexShrink={0} gap={2} isDisabled={!traceTaperEnds} overflow="hidden">
          <FormLabel m={0} mt={1} whiteSpace="nowrap">
            {t('controlLayers.vectorEdit.taper')}
          </FormLabel>
          <PopoverAnchor>
            <NumberInput
              size="sm"
              variant="outline"
              display="flex"
              alignItems="center"
              min={1}
              max={500}
              step={10}
              value={localTaper}
              onChange={onTaperChange}
              onBlur={commitTaper}
              onKeyDown={onTaperKeyDown}
              clampValueOnBlur={false}
              format={formatPercent}
              w={28}
            >
              <PopoverTrigger>
                <NumberInputField
                  _focusVisible={{ zIndex: 0 }}
                  aria-label={t('controlLayers.vectorEdit.taper')}
                  title=""
                  onFocus={selectInputValue}
                />
              </PopoverTrigger>
              <NumberInputStepper>
                <NumberIncrementStepper>
                  <MdKeyboardArrowUp />
                </NumberIncrementStepper>
                <NumberDecrementStepper>
                  <MdKeyboardArrowDown />
                </NumberDecrementStepper>
              </NumberInputStepper>
            </NumberInput>
          </PopoverAnchor>
        </FormControl>
        <Portal>
          <PopoverContent w={200} pt={0} pb={2} px={4}>
            <PopoverArrow />
            <PopoverBody>
              <CompositeSlider
                min={1}
                max={500}
                value={Number.isFinite(localTaper) ? localTaper : traceTaper}
                onChange={onTaperSliderChange}
                defaultValue={100}
                marks={taperMarks}
                formatValue={String}
                alwaysShowMarks
              />
            </PopoverBody>
          </PopoverContent>
        </Portal>
      </Popover>
    </Fragment>
  );
});

VectorLayerTraceWidth.displayName = 'VectorLayerTraceWidth';
