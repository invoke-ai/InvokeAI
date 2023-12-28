import type {
  ChakraProps,
  SliderProps as ChakraSliderProps,
} from '@chakra-ui/react';

export type InvSliderProps = Omit<ChakraSliderProps, 'value'> & {
  /**
   * The value
   */
  value: number;
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
  onChange: (v: number) => void;
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
  /**
   * Whether or not to render a number input
   */
  withNumberInput?: boolean;
  /**
   * The number input min (defaults to the slider min)
   */
  numberInputMin?: number;
  /**
   * The number input max (defaults to the slider max)
   */
  numberInputMax?: number;
  /**
   * The width of the number input
   */
  numberInputWidth?: ChakraProps['width'];
};

export type InvFormattedMark = { value: number; label: string };

export type InvSliderMarkProps = {
  value: number;
  label: string;
  index: number;
  total: number;
};
