import {
  FormControl,
  FormLabel,
  HStack,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Slider,
  SliderFilledTrack,
  SliderMark,
  SliderThumb,
  SliderTrack,
  Tooltip,
} from '@chakra-ui/react';
import React, { FocusEvent, useEffect, useState } from 'react';
import { BiReset } from 'react-icons/bi';
import IAIIconButton from './IAIIconButton';
import _ from 'lodash';

const numberStringRegex = /^-?(0\.)?\.?$/;

interface IAIFullSliderProps {
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
  tooltipSuffix?: string;
  hideTooltip?: boolean;
  styleClass?: string;
}

export default function IAIFullSlider(props: IAIFullSliderProps) {
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
    inputWidth = '5rem',
    inputReadOnly = false,
    withReset = false,
    hideTooltip = false,
    handleReset,
    isResetDisabled,
    styleClass,
  } = props;

  const [localInputValue, setLocalInputValue] = useState<string>(String(value));

  useEffect(() => {
    if (String(value) !== localInputValue && localInputValue !== '') {
      setLocalInputValue(String(value));
    }
  }, [value, setLocalInputValue]);

  const handleInputBlur = (e: FocusEvent<HTMLInputElement>) => {
    const clamped = _.clamp(
      isInteger ? Math.floor(Number(e.target.value)) : Number(e.target.value),
      min,
      max
    );
    setLocalInputValue(String(clamped));
    onChange(clamped);
  };

  const handleInputChange = (v: any) => {
    setLocalInputValue(v);
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
    >
      <FormLabel className="invokeai__slider-component-label">
        {label}
      </FormLabel>

      <HStack w={'100%'} gap={1}>
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
        >
          {withSliderMarks && (
            <>
              <SliderMark
                value={min}
                className="invokeai__slider-mark invokeai__slider-mark-start"
              >
                {min}
              </SliderMark>
              <SliderMark
                value={max}
                className="invokeai__slider-mark invokeai__slider-mark-end"
              >
                {max}
              </SliderMark>
            </>
          )}

          <SliderTrack className="invokeai__slider_track">
            <SliderFilledTrack className="invokeai__slider_track-filled" />
          </SliderTrack>

          <Tooltip
            hasArrow
            className="invokeai__slider-component-tooltip"
            placement="top"
            isOpen={showTooltip}
            label={`${value}${tooltipSuffix}`}
            hidden={hideTooltip}
          >
            <SliderThumb className="invokeai__slider-thumb" />
          </Tooltip>
        </Slider>

        {withInput && (
          <NumberInput
            min={min}
            max={max}
            step={step}
            value={localInputValue}
            onChange={handleInputChange}
            onBlur={handleInputBlur}
            className="invokeai__slider-number-field"
          >
            <NumberInputField
              className="invokeai__slider-number-input"
              width={inputWidth}
              readOnly={inputReadOnly}
            />
            <NumberInputStepper>
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
          />
        )}
      </HStack>
    </FormControl>
  );
}
