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
import React, { FocusEvent, useEffect, useMemo, useState } from 'react';
import { BiReset } from 'react-icons/bi';
import IAIIconButton, { IAIIconButtonProps } from './IAIIconButton';
import _ from 'lodash';

export type IAIFullSliderProps = {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
  withSliderMarks?: boolean;
  sliderMarkLeftOffset?: number;
  sliderMarkRightOffset?: number;
  withInput?: boolean;
  isInteger?: boolean;
  width?: string | number;
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
  styleClass?: string;
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
    width = '100%',
    tooltipSuffix = '',
    withSliderMarks = false,
    sliderMarkLeftOffset = 0,
    sliderMarkRightOffset = -7,
    withInput = false,
    isInteger = false,
    inputWidth = '5.5rem',
    inputReadOnly = true,
    withReset = false,
    hideTooltip = false,
    isCompact = false,
    handleReset,
    isResetDisabled,
    isSliderDisabled,
    isInputDisabled,
    styleClass,
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

  const [localInputValue, setLocalInputValue] = useState<string>(String(value));

  const numberInputMax = useMemo(
    () => (sliderNumberInputProps?.max ? sliderNumberInputProps.max : max),
    [max, sliderNumberInputProps?.max]
  );

  useEffect(() => {
    if (String(value) !== localInputValue && localInputValue !== '') {
      setLocalInputValue(String(value));
    }
  }, [value, localInputValue, setLocalInputValue]);

  const handleInputBlur = (e: FocusEvent<HTMLInputElement>) => {
    const clamped = _.clamp(
      isInteger ? Math.floor(Number(e.target.value)) : Number(e.target.value),
      min,
      numberInputMax
    );
    setLocalInputValue(String(clamped));
    onChange(clamped);
  };

  const handleInputChange = (v: number | string) => {
    setLocalInputValue(String(v));
    onChange(Number(v));
  };

  const handleResetDisable = () => {
    if (!handleReset) return;
    handleReset();
  };

  return (
    <FormControl
      className={
        styleClass
          ? `invokeai__slider-component ${styleClass}`
          : `invokeai__slider-component`
      }
      data-markers={withSliderMarks}
      style={
        isCompact
          ? {
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              columnGap: '1rem',
              margin: 0,
              padding: 0,
            }
          : {}
      }
      {...sliderFormControlProps}
    >
      <FormLabel
        className="invokeai__slider-component-label"
        {...sliderFormLabelProps}
      >
        {label}
      </FormLabel>

      <HStack w={'100%'} gap={2} alignItems="center">
        <Slider
          aria-label={label}
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={handleInputChange}
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
          focusThumbOnChange={false}
          isDisabled={isSliderDisabled}
          width={width}
          {...rest}
        >
          {withSliderMarks && (
            <>
              <SliderMark
                value={min}
                className="invokeai__slider-mark invokeai__slider-mark-start"
                ml={sliderMarkLeftOffset}
                {...sliderMarkProps}
              >
                {min}
              </SliderMark>
              <SliderMark
                value={max}
                className="invokeai__slider-mark invokeai__slider-mark-end"
                ml={sliderMarkRightOffset}
                {...sliderMarkProps}
              >
                {max}
              </SliderMark>
            </>
          )}

          <SliderTrack className="invokeai__slider_track" {...sliderTrackProps}>
            <SliderFilledTrack className="invokeai__slider_track-filled" />
          </SliderTrack>

          <Tooltip
            hasArrow
            className="invokeai__slider-component-tooltip"
            placement="top"
            isOpen={showTooltip}
            label={`${value}${tooltipSuffix}`}
            hidden={hideTooltip}
            {...sliderTooltipProps}
          >
            <SliderThumb
              className="invokeai__slider-thumb"
              {...sliderThumbProps}
            />
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
            className="invokeai__slider-number-field"
            isDisabled={isInputDisabled}
            {...sliderNumberInputProps}
          >
            <NumberInputField
              className="invokeai__slider-number-input"
              width={inputWidth}
              minWidth={inputWidth}
              readOnly={inputReadOnly}
              {...sliderNumberInputFieldProps}
            />
            <NumberInputStepper {...sliderNumberInputStepperProps}>
              <NumberIncrementStepper className="invokeai__slider-number-stepper" />
              <NumberDecrementStepper className="invokeai__slider-number-stepper" />
            </NumberInputStepper>
          </NumberInput>
        )}

        {withReset && (
          <IAIIconButton
            size={'sm'}
            aria-label={'Reset'}
            tooltip={'Reset'}
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
