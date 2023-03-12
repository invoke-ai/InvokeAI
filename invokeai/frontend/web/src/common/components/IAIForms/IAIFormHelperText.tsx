import { FormHelperText, FormHelperTextProps } from '@chakra-ui/react';
import { ReactNode } from 'react';

type IAIFormHelperTextProps = FormHelperTextProps & {
  children: ReactNode | string;
};

export default function IAIFormHelperText(props: IAIFormHelperTextProps) {
  const { children, ...rest } = props;
  return (
    <FormHelperText margin={0} color="base.400" {...rest}>
      {children}
    </FormHelperText>
  );
}
