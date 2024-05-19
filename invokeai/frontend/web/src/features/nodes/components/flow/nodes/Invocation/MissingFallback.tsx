import { useDoesFieldExist } from 'features/nodes/hooks/useDoesFieldExist';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName?: string;
}>;

export const MissingFallback = memo((props: Props) => {
  // We must be careful here to avoid race conditions where a deleted node is still referenced as an exposed field
  const exists = useDoesFieldExist(props.nodeId, props.fieldName);
  if (!exists) {
    return null;
  }

  return props.children;
});

MissingFallback.displayName = 'MissingFallback';
