import { Tooltip } from '@chakra-ui/react';
import * as Slider from '@radix-ui/react-slider';
import React from 'react';
import IAITooltip from './IAITooltip';

type IAISliderProps = Slider.SliderProps & {
  value: number[];
  tooltipLabel?: string;
  orientation?: 'horizontal' | 'vertial';
  trackProps?: Slider.SliderTrackProps;
  rangeProps?: Slider.SliderRangeProps;
  thumbProps?: Slider.SliderThumbProps;
};

const _IAISlider = (props: IAISliderProps) => {
  const {
    value,
    tooltipLabel,
    orientation,
    trackProps,
    rangeProps,
    thumbProps,
    ...rest
  } = props;
  return (
    <Slider.Root
      className="invokeai__slider-root"
      {...rest}
      data-orientation={orientation || 'horizontal'}
    >
      <Slider.Track {...trackProps} className="invokeai__slider-track">
        <Slider.Range {...rangeProps} className="invokeai__slider-range" />
      </Slider.Track>
      <Tooltip label={tooltipLabel ?? value[0]} placement="top">
        <Slider.Thumb {...thumbProps} className="invokeai__slider-thumb">
          <div className="invokeai__slider-thumb-div" />
          {/*<IAITooltip trigger={<div className="invokeai__slider-thumb-div" />}>
          {value && value[0]}
        </IAITooltip>*/}
        </Slider.Thumb>
      </Tooltip>
    </Slider.Root>
  );
};

export default _IAISlider;
