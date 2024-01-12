import type { PropsWithChildren } from 'react';

export type InvSingleAccordionProps = PropsWithChildren<{
  label: string;
  badges?: (string | number)[];
  defaultIsOpen?: boolean;
  onClick?: (isOpen?: boolean) => void
}>;
