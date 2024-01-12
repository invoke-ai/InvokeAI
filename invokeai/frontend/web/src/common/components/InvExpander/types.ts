import { ThunkAction } from '@reduxjs/toolkit';
import type { PropsWithChildren } from 'react';

export type InvExpanderProps = PropsWithChildren<{
  label?: string;
  defaultIsOpen?: boolean;
  onClick?: (isOpen: boolean) => void
}>;
