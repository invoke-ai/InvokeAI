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
import { clamp } from 'lodash';

import { FocusEvent, useEffect, useMemo, useState } from 'react';
import { BiReset } from 'react-icons/bi';
import IAIIconButton, { IAIIconButtonProps } from './IAIIconButton';

export type IAIFullSliderProps = {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
  withSliderMarks?: boolean;
  withInput?: boolean;
  isInteger?: boolean;
  inputWidth?: string | number;
  inputReadOnly?: boolean;
  withReset?: boolean;
  handleReset?: () => void;
  isResetDisabled?: boolean;
  isSliderDisabled?: boolean;
  isInputDisabled?: boolean;
  tooltipSuffix?: string;
  hideTooltip?: boolean;
  isCompact?: boolean;
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

export default function IAISlider(props: IAIFullSliderProps) {
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
    inputReadOnly = false,
    withReset = false,
    hideTooltip = false,
    isCompact = false,
    handleReset,
    isResetDisabled,
    isSliderDisabled,
    isInputDisabled,
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

  const [localInputValue, setLocalInputValue] = useState<
    string | number | undefined
  >(String(value));

  useEffect(() => {
    setLocalInputValue(value);
  }, [value]);

  const numberInputMax = useMemo(
    () => (sliderNumberInputProps?.max ? sliderNumberInputProps.max : max),
    [max, sliderNumberInputProps?.max]
  );

  const handleSliderChange = (v: number) => {
    onChange(v);
  };

  const handleInputBlur = (e: FocusEvent<HTMLInputElement>) => {
    if (e.target.value === '') e.target.value = String(min);
    const clamped = clamp(
      isInteger ? Math.floor(Number(e.target.value)) : Number(localInputValue),
      min,
      numberInputMax
    );
    onChange(clamped);
  };

  const handleInputChange = (v: number | string) => {
    setLocalInputValue(v);
  };

  const handleResetDisable = () => {
    if (!handleReset) return;
    handleReset();
  };

  return (
    <FormControl
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
      {...sliderFormControlProps}
    >
      <FormLabel {...sliderFormLabelProps} mb={-1}>
        {label}
      </FormLabel>

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
          isDisabled={isSliderDisabled}
          // width={width}
          {...rest}
        >
          {withSliderMarks && (
            <>
              <SliderMark
                value={min}
                insetInlineStart={0}
                sx={{ insetInlineStart: 'unset !important' }}
                {...sliderMarkProps}
              >
                {min}
              </SliderMark>
              <SliderMark
                value={max}
                insetInlineEnd={0}
                sx={{ insetInlineStart: 'unset !important' }}
                {...sliderMarkProps}
              >
                {max}
              </SliderMark>
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
            <SliderThumb {...sliderThumbProps} />
          </Tooltip>
        </Slider>

        {withInput && (
          <NumberInput
            min={min}
            max={numberInputMax}
            step={step}
            value={localInputValue}
            onChange={handleInputChange}
            onBlur={handleInputBlur}
            isDisabled={isInputDisabled}
            {...sliderNumberInputProps}
          >
            <NumberInputField
              readOnly={inputReadOnly}
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
            aria-label="Reset"
            tooltip="Reset"
            icon={<BiReset />}
            onClick={handleResetDisable}
            isDisabled={isResetDisabled}
            {...sliderIAIIconButtonProps}
          />
        )}
      </HStack>
    </FormControl>
  );
}
