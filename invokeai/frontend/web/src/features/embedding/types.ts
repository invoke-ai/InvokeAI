import type { PropsWithChildren } from 'react';

export type EmbeddingSelectProps = {
  onSelect: (v: string) => void;
  onClose: () => void;
};

export type EmbeddingPopoverProps = PropsWithChildren &
  EmbeddingSelectProps & {
    isOpen: boolean;
    width?: number | string;
  };
