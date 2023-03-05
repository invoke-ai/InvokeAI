import {
  FormControl,
  FormControlProps,
  FormLabel,
  Input,
  InputProps,
} from '@chakra-ui/react';
import { ChangeEvent } from 'react';

interface IAIInputProps extends InputProps {
  label?: string;
  value?: string;
  size?: string;
  onChange?: (e: ChangeEvent<HTMLInputElement>) => void;
  formControlProps?: Omit<FormControlProps, 'isInvalid' | 'isDisabled'>;
}

export default function IAIInput(props: IAIInputProps) {
  const {
    label = '',
    isDisabled = false,
    isInvalid,
    formControlProps,
    ...rest
  } = props;

  return (
    <FormControl
      isInvalid={isInvalid}
      isDisabled={isDisabled}
      {...formControlProps}
    >
      {label !== '' && <FormLabel>{label}</FormLabel>}
      <Input {...rest} />
    </FormControl>
  );
}
