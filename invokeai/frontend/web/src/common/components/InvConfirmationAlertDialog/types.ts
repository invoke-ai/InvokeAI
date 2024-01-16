import type { AlertDialogProps } from '@chakra-ui/react';
import type { PropsWithChildren } from 'react';

export type InvConfirmationAlertDialogProps = Omit<
  AlertDialogProps,
  'leastDestructiveRef' | 'isOpen' | 'onClose'
> &
  PropsWithChildren<{
    isOpen: boolean;
    onClose: () => void;
    acceptButtonText?: string;
    acceptCallback: () => void;
    cancelButtonText?: string;
    cancelCallback?: () => void;
    title: string;
  }>;
