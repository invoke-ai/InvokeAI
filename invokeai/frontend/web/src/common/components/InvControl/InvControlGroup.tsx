import type { FormControlProps, FormLabelProps } from '@chakra-ui/react';
import type { PropsWithChildren } from 'react';
import { createContext, memo } from 'react';

export type InvControlGroupProps = {
  labelProps?: FormLabelProps;
  controlProps?: FormControlProps;
  isDisabled?: boolean;
  orientation?: 'horizontal' | 'vertical';
};

export const InvControlGroupContext = createContext<InvControlGroupProps>({});

export const InvControlGroup = memo(
  ({ children, ...context }: PropsWithChildren<InvControlGroupProps>) => {
    return (
      <InvControlGroupContext.Provider value={context}>
        {children}
      </InvControlGroupContext.Provider>
    );
  }
);

InvControlGroup.displayName = 'InvControlGroup';
