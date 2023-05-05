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
  horizontal?: boolean;
  spaceEvenly?: boolean;
};
/**
 * Customized Chakra FormControl + Select multi-part component.
 */
const IAISelect = (props: IAISelectProps) => {
  const {
    label,
    isDisabled,
    validValues,
    tooltip,
    tooltipProps,
    horizontal,
    spaceEvenly,
    ...rest
  } = props;
  return (
    <FormControl
      isDisabled={isDisabled}
      onClick={(e: MouseEvent<HTMLDivElement>) => {
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
        e.nativeEvent.stopPropagation();
        e.nativeEvent.cancelBubble = true;
      }}
      sx={
        horizontal
          ? {
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 4,
            }
          : {}
      }
    >
      {label && (
        <FormLabel sx={spaceEvenly ? { flexBasis: 0, flexGrow: 1 } : {}}>
          {label}
        </FormLabel>
      )}
      <Tooltip label={tooltip} {...tooltipProps}>
        <Select
          {...rest}
          rootProps={{ sx: spaceEvenly ? { flexBasis: 0, flexGrow: 1 } : {} }}
        >
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
