import { FormControl, FormLabel, Select, SelectProps } from '@chakra-ui/react';
import { MouseEvent } from 'react';

type IAISelectProps = SelectProps & {
  label: string;
  styleClass?: string;
  validValues:
    | Array<number | string>
    | Array<{ key: string; value: string | number }>;
};
/**
 * Customized Chakra FormControl + Select multi-part component.
 */
const IAISelect = (props: IAISelectProps) => {
  const {
    label,
    isDisabled,
    validValues,
    size = 'sm',
    fontSize = 'md',
    styleClass,
    ...rest
  } = props;
  return (
    <FormControl
      isDisabled={isDisabled}
      className={`invokeai__select ${styleClass}`}
      onClick={(e: MouseEvent<HTMLDivElement>) => {
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
        e.nativeEvent.stopPropagation();
        e.nativeEvent.cancelBubble = true;
      }}
    >
      <FormLabel
        className="invokeai__select-label"
        fontSize={fontSize}
        marginBottom={1}
        flexGrow={2}
        whiteSpace="nowrap"
      >
        {label}
      </FormLabel>
      <Select
        className="invokeai__select-picker"
        fontSize={fontSize}
        size={size}
        {...rest}
      >
        {validValues.map((opt) => {
          return typeof opt === 'string' || typeof opt === 'number' ? (
            <option key={opt} value={opt} className="invokeai__select-option">
              {opt}
            </option>
          ) : (
            <option
              key={opt.value}
              value={opt.value}
              className="invokeai__select-option"
            >
              {opt.key}
            </option>
          );
        })}
      </Select>
    </FormControl>
  );
};

export default IAISelect;
