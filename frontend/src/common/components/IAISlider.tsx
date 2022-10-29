import {
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  FormControl,
  FormLabel,
  Tooltip,
  SliderProps,
  FormControlProps,
  FormLabelProps,
  SliderTrackProps,
  SliderThumbProps,
  TooltipProps,
  SliderInnerTrackProps,
} from '@chakra-ui/react';

type IAISliderProps = SliderProps & {
  label?: string;
  styleClass?: string;
  formControlProps?: FormControlProps;
  formLabelProps?: FormLabelProps;
  sliderTrackProps?: SliderTrackProps;
  sliderInnerTrackProps?: SliderInnerTrackProps;
  sliderThumbProps?: SliderThumbProps;
  sliderThumbTooltipProps?: Omit<TooltipProps, 'children'>;
};

const IAISlider = (props: IAISliderProps) => {
  const {
    label,
    styleClass,
    formControlProps,
    formLabelProps,
    sliderTrackProps,
    sliderInnerTrackProps,
    sliderThumbProps,
    sliderThumbTooltipProps,
    ...rest
  } = props;
  return (
    <FormControl
      className={`invokeai__slider-form-control ${styleClass}`}
      {...formControlProps}
    >
      <div className="invokeai__slider-inner-container">
        <FormLabel
          className={`invokeai__slider-form-label`}
          whiteSpace="nowrap"
          {...formLabelProps}
        >
          {label}
        </FormLabel>

        <Slider
          className={`invokeai__slider-root`}
          aria-label={label}
          {...rest}
        >
          <SliderTrack
            className={`invokeai__slider-track`}
            {...sliderTrackProps}
          >
            <SliderFilledTrack
              className={`invokeai__slider-filled-track`}
              {...sliderInnerTrackProps}
            />
          </SliderTrack>

          <Tooltip
            className={`invokeai__slider-thumb-tooltip`}
            placement="top"
            hasArrow
            {...sliderThumbTooltipProps}
          >
            <SliderThumb
              className={`invokeai__slider-thumb`}
              {...sliderThumbProps}
            />
          </Tooltip>
        </Slider>
      </div>
    </FormControl>
  );
};

export default IAISlider;
