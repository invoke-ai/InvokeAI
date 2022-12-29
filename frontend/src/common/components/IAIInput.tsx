import { FormControl, FormLabel, Input, InputProps } from '@chakra-ui/react';
import { ChangeEvent } from 'react';

interface IAIInputProps extends InputProps {
  styleClass?: string;
  label?: string;
  width?: string | number;
  value?: string;
  size?: string;
  onChange?: (e: ChangeEvent<HTMLInputElement>) => void;
}

export default function IAIInput(props: IAIInputProps) {
  const {
    label = '',
    styleClass,
    isDisabled = false,
    fontSize = 'sm',
    width,
    size = 'sm',
    isInvalid,
    ...rest
  } = props;

  return (
    <FormControl
      className={`input ${styleClass}`}
      isInvalid={isInvalid}
      isDisabled={isDisabled}
    >
      {label !== '' && (
        <FormLabel
          fontSize={fontSize}
          fontWeight="bold"
          alignItems="center"
          whiteSpace="nowrap"
          marginBottom={0}
          marginRight={0}
          className="input-label"
        >
          {label}
        </FormLabel>
      )}
      <Input {...rest} className="input-entry" size={size} width={width} />
    </FormControl>
  );
}
