import {
  FormControl,
  FormControlProps,
  FormLabel,
  Input,
  InputProps,
} from '@chakra-ui/react';
import { stopPastePropagation } from 'common/util/stopPastePropagation';
import { ChangeEvent, memo } from 'react';

interface IAIInputProps extends InputProps {
  label?: string;
  value?: string;
  size?: string;
  onChange?: (e: ChangeEvent<HTMLInputElement>) => void;
  formControlProps?: Omit<FormControlProps, 'isInvalid' | 'isDisabled'>;
}

const IAIInput = (props: IAIInputProps) => {
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
      <Input {...rest} onPaste={stopPastePropagation} />
    </FormControl>
  );
};

export default memo(IAIInput);
