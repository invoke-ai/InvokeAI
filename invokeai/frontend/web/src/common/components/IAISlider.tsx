import {
  FormControl,
  FormControlProps,
  FormLabel,
  FormLabelProps,
  HStack,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputFieldProps,
  NumberInputProps,
  NumberInputStepper,
  NumberInputStepperProps,
  Slider,
  SliderFilledTrack,
  SliderMark,
  SliderMarkProps,
  SliderThumb,
  SliderThumbProps,
  SliderTrack,
  SliderTrackProps,
  Tooltip,
  TooltipProps,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { roundDownToMultiple } from 'common/util/roundDownToMultiple';
import { shiftKeyPressed } from 'features/ui/store/hotkeysSlice';
import { clamp } from 'lodash-es';
import {
  FocusEvent,
  KeyboardEvent,
  MouseEvent,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import { BiReset } from 'react-icons/bi';
import IAIIconButton, { IAIIconButtonProps } from './IAIIconButton';

export type IAIFullSliderProps = {
  label?: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
  withSliderMarks?: boolean;
  withInput?: boolean;
  isInteger?: boolean;
  inputWidth?: string | number;
  withReset?: boolean;
  handleReset?: () => void;
  tooltipSuffix?: string;
  hideTooltip?: boolean;
  isCompact?: boolean;
  isDisabled?: boolean;
  sliderMarks?: number[];
  sliderFormControlProps?: FormControlProps;
  sliderFormLabelProps?: FormLabelProps;
  sliderMarkProps?: Omit<SliderMarkProps, 'value'>;
  sliderTrackProps?: SliderTrackProps;
  sliderThumbProps?: SliderThumbProps;
  sliderNumberInputProps?: NumberInputProps;
  sliderNumberInputFieldProps?: NumberInputFieldProps;
  sliderNumberInputStepperProps?: NumberInputStepperProps;
  sliderTooltipProps?: Omit<TooltipProps, 'children'>;
  sliderIAIIconButtonProps?: IAIIconButtonProps;
};

const IAISlider = (props: IAIFullSliderProps) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const {
    label,
    value,
    min = 1,
    max = 100,
    step = 1,
    onChange,
    tooltipSuffix = '',
    withSliderMarks = false,
    withInput = false,
    isInteger = false,
    inputWidth = 16,
    withReset = false,
    hideTooltip = false,
    isCompact = false,
    isDisabled = false,
    sliderMarks,
    handleReset,
    sliderFormControlProps,
    sliderFormLabelProps,
    sliderMarkProps,
    sliderTrackProps,
    sliderThumbProps,
    sliderNumberInputProps,
    sliderNumberInputFieldProps,
    sliderNumberInputStepperProps,
    sliderTooltipProps,
    sliderIAIIconButtonProps,
    ...rest
  } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const [localInputValue, setLocalInputValue] = useState<
    string | number | undefined
  >(String(value));

  useEffect(() => {
    setLocalInputValue(value);
  }, [value]);

  const numberInputMin = useMemo(
    () => (sliderNumberInputProps?.min ? sliderNumberInputProps.min : min),
    [min, sliderNumberInputProps?.min]
  );

  const numberInputMax = useMemo(
    () => (sliderNumberInputProps?.max ? sliderNumberInputProps.max : max),
    [max, sliderNumberInputProps?.max]
  );

  const handleSliderChange = useCallback(
    (v: number) => {
      onChange(v);
    },
    [onChange]
  );

  const handleInputBlur = useCallback(
    (e: FocusEvent<HTMLInputElement>) => {
      if (e.target.value === '') {
        e.target.value = String(numberInputMin);
      }
      const clamped = clamp(
        isInteger
          ? Math.floor(Number(e.target.value))
          : Number(localInputValue),
        numberInputMin,
        numberInputMax
      );
      const quantized = roundDownToMultiple(clamped, step);
      onChange(quantized);
      setLocalInputValue(quantized);
    },
    [isInteger, localInputValue, numberInputMin, numberInputMax, onChange, step]
  );

  const handleInputChange = useCallback((v: number | string) => {
    setLocalInputValue(v);
  }, []);

  const handleResetDisable = useCallback(() => {
    if (!handleReset) {
      return;
    }
    handleReset();
  }, [handleReset]);

  const forceInputBlur = useCallback((e: MouseEvent) => {
    if (e.target instanceof HTMLDivElement) {
      e.target.focus();
    }
  }, []);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.shiftKey) {
        dispatch(shiftKeyPressed(true));
      }
    },
    [dispatch]
  );

  const handleKeyUp = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (!e.shiftKey) {
        dispatch(shiftKeyPressed(false));
      }
    },
    [dispatch]
  );

  return (
    <FormControl
      onClick={forceInputBlur}
      sx={
        isCompact
          ? {
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              columnGap: 4,
              margin: 0,
              padding: 0,
            }
          : {}
      }
      isDisabled={isDisabled}
      {...sliderFormControlProps}
    >
      {label && (
        <FormLabel sx={withInput ? { mb: -1.5 } : {}} {...sliderFormLabelProps}>
          {label}
        </FormLabel>
      )}

      <HStack w="100%" gap={2} alignItems="center">
        <Slider
          aria-label={label}
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={handleSliderChange}
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
          focusThumbOnChange={false}
          isDisabled={isDisabled}
          {...rest}
        >
          {withSliderMarks && !sliderMarks && (
            <>
              <SliderMark
                value={min}
                sx={{
                  insetInlineStart: '0 !important',
                  insetInlineEnd: 'unset !important',
                }}
                {...sliderMarkProps}
              >
                {min}
              </SliderMark>
              <SliderMark
                value={max}
                sx={{
                  insetInlineStart: 'unset !important',
                  insetInlineEnd: '0 !important',
                }}
                {...sliderMarkProps}
              >
                {max}
              </SliderMark>
            </>
          )}
          {withSliderMarks && sliderMarks && (
            <>
              {sliderMarks.map((m, i) => {
                if (i === 0) {
                  return (
                    <SliderMark
                      key={m}
                      value={m}
                      sx={{
                        insetInlineStart: '0 !important',
                        insetInlineEnd: 'unset !important',
                      }}
                      {...sliderMarkProps}
                    >
                      {m}
                    </SliderMark>
                  );
                } else if (i === sliderMarks.length - 1) {
                  return (
                    <SliderMark
                      key={m}
                      value={m}
                      sx={{
                        insetInlineStart: 'unset !important',
                        insetInlineEnd: '0 !important',
                      }}
                      {...sliderMarkProps}
                    >
                      {m}
                    </SliderMark>
                  );
                } else {
                  return (
                    <SliderMark
                      key={m}
                      value={m}
                      sx={{
                        transform: 'translateX(-50%)',
                      }}
                      {...sliderMarkProps}
                    >
                      {m}
                    </SliderMark>
                  );
                }
              })}
            </>
          )}

          <SliderTrack {...sliderTrackProps}>
            <SliderFilledTrack />
          </SliderTrack>

          <Tooltip
            hasArrow
            placement="top"
            isOpen={showTooltip}
            label={`${value}${tooltipSuffix}`}
            hidden={hideTooltip}
            {...sliderTooltipProps}
          >
            <SliderThumb {...sliderThumbProps} zIndex={0} />
          </Tooltip>
        </Slider>

        {withInput && (
          <NumberInput
            min={numberInputMin}
            max={numberInputMax}
            step={step}
            value={localInputValue}
            onChange={handleInputChange}
            onBlur={handleInputBlur}
            focusInputOnChange={false}
            {...sliderNumberInputProps}
          >
            <NumberInputField
              onKeyDown={handleKeyDown}
              onKeyUp={handleKeyUp}
              minWidth={inputWidth}
              {...sliderNumberInputFieldProps}
            />
            <NumberInputStepper {...sliderNumberInputStepperProps}>
              <NumberIncrementStepper
                onClick={() => onChange(Number(localInputValue))}
              />
              <NumberDecrementStepper
                onClick={() => onChange(Number(localInputValue))}
              />
            </NumberInputStepper>
          </NumberInput>
        )}

        {withReset && (
          <IAIIconButton
            size="sm"
            aria-label={t('accessibility.reset')}
            tooltip={t('accessibility.reset')}
            icon={<BiReset />}
            isDisabled={isDisabled}
            onClick={handleResetDisable}
            {...sliderIAIIconButtonProps}
          />
        )}
      </HStack>
    </FormControl>
  );
};

export default memo(IAISlider);
