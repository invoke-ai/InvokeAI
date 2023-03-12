import {
  FormControl,
  FormLabel,
  Select,
  SelectProps,
  Tooltip,
  TooltipProps,
} from '@chakra-ui/react';
import { memo, MouseEvent } from 'react';
import IAIOption from './IAIOption';

type IAISelectProps = SelectProps & {
  label?: string;
  tooltip?: string;
  tooltipProps?: Omit<TooltipProps, 'children'>;
  validValues:
    | Array<number | string>
    | Array<{ key: string; value: string | number }>;
};
/**
 * Customized Chakra FormControl + Select multi-part component.
 */
const IAISelect = (props: IAISelectProps) => {
  const { label, isDisabled, validValues, tooltip, tooltipProps, ...rest } =
    props;
  return (
    <FormControl
      isDisabled={isDisabled}
      onClick={(e: MouseEvent<HTMLDivElement>) => {
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
        e.nativeEvent.stopPropagation();
        e.nativeEvent.cancelBubble = true;
      }}
    >
      {label && <FormLabel>{label}</FormLabel>}
      <Tooltip label={tooltip} {...tooltipProps}>
        <Select {...rest}>
          {validValues.map((opt) => {
            return typeof opt === 'string' || typeof opt === 'number' ? (
              <IAIOption key={opt} value={opt}>
                {opt}
              </IAIOption>
            ) : (
              <IAIOption key={opt.value} value={opt.value}>
                {opt.key}
              </IAIOption>
            );
          })}
        </Select>
      </Tooltip>
    </FormControl>
  );
};

export default memo(IAISelect);
