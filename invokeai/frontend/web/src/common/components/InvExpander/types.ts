import type { PropsWithChildren } from 'react';

export type InvExpanderProps = PropsWithChildren<{
  label?: string;
  defaultIsOpen?: boolean;
  id?: string;
}>;
