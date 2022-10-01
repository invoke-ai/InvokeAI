import {
  FormControl,
  FormLabel,
  HStack,
  Input,
  InputProps,
} from '@chakra-ui/react';
import { ChangeEvent } from 'react';

interface IAIInputProps extends InputProps {
  styleClass?: string;
  label?: string;
  width?: string | number;
  value: string;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
}

export default function IAIInput(props: IAIInputProps) {
  const {
    label,
    styleClass,
    isDisabled = false,
    fontSize = '1rem',
    size = 'sm',
    width = '150px',
    textAlign,
    isInvalid,
    ...rest
  } = props;

  return (
    <FormControl
      className={`input ${styleClass}`}
      isInvalid={isInvalid}
      isDisabled={isDisabled}
      flexGrow={1}
    >
      <HStack>
        <FormLabel
          fontSize={fontSize}
          marginBottom={1}
          whiteSpace="nowrap"
          className="input-label"
        >
          {label}
        </FormLabel>
        <Input {...rest} className="input-entry" size={'sm'} width="12rem" />
      </HStack>
    </FormControl>
  );
}
