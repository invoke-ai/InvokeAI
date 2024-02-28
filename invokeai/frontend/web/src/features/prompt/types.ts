import type { PropsWithChildren } from 'react';

export type PromptTriggerSelectProps = {
  onSelect: (v: string) => void;
  onClose: () => void;
};

export type PromptPopoverProps = PropsWithChildren &
  PromptTriggerSelectProps & {
    isOpen: boolean;
    width?: number | string;

  };
