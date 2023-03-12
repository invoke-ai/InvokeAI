import { FormErrorMessage, FormErrorMessageProps } from '@chakra-ui/react';
import { ReactNode } from 'react';

type IAIFormErrorMessageProps = FormErrorMessageProps & {
  children: ReactNode | string;
};

export default function IAIFormErrorMessage(props: IAIFormErrorMessageProps) {
  const { children, ...rest } = props;
  return (
    <FormErrorMessage color="error.400" {...rest}>
      {children}
    </FormErrorMessage>
  );
}
