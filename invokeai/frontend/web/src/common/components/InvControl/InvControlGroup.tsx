import type { FormLabelProps } from '@chakra-ui/react';
import type { PropsWithChildren } from 'react';
import { createContext } from 'react';

export type InvControlGroupProps = {
  labelProps?: FormLabelProps;
  isDisabled?: boolean;
  orientation?: 'horizontal' | 'vertical';
};

export const InvControlGroupContext = createContext<InvControlGroupProps>({});

export const InvControlGroup = ({
  children,
  ...context
}: PropsWithChildren<InvControlGroupProps>) => {
  return (
    <InvControlGroupContext.Provider value={context}>
      {children}
    </InvControlGroupContext.Provider>
  );
};
