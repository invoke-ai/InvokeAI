import type { RangeSliderProps as ChakraRangeSliderProps } from '@chakra-ui/react';
import type { InvSliderMarkProps } from 'common/components/InvSlider/types';

export type InvRangeSliderProps = Omit<ChakraRangeSliderProps, 'value'> & {
  /**
   * The value
   */
  value: [number, number];
  /**
   * The minimum value
   */
  min: number;
  /**
   * The maximum value
   */
  max: number;
  /**
   * The default step
   */
  step?: number;
  /**
   * The fine step (when shift is pressed)
   */
  fineStep?: number;
  /**
   * The change handler
   */
  onChange: (v: [number, number]) => void;
  /**
   * The reset handler, called on double-click of the thumb
   */
  onReset?: () => void;
  /**
   * The value formatter
   */
  formatValue?: (v: number) => string;
  /**
   * Whether the slider is disabled
   */
  isDisabled?: boolean;
  /**
   * The marks to render below the slider. If true, will use the min and max values.
   */
  marks?: number[] | true;
  /**
   * Whether to show a tooltip over the slider thumb
   */
  withThumbTooltip?: boolean;
};

export type InvRangeSliderMarkProps = InvSliderMarkProps;
