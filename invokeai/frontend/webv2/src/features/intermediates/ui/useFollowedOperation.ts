import type { RefObject } from 'react';

import {
  activeOperationStore,
  attachIntermediatesManager,
  followIntermediatesOperation,
} from '@features/intermediates/data/operationStore';
import { intermediatesOperationQueryOptions } from '@features/intermediates/data/queries';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';

/** The operation the section follows, kept across visits by the operation store, which this hook subscribes to. */
export const useFollowedOperation = ({ fallbackFocusRef }: { fallbackFocusRef: RefObject<HTMLElement | null> }) => {
  const queryClient = useQueryClient();
  const operationId = activeOperationStore.useSelector((snapshot) => snapshot.operationId);
  const query = useQuery({ ...intermediatesOperationQueryOptions(operationId ?? ''), enabled: operationId !== null });

  useMountEffect(() => attachIntermediatesManager(queryClient));

  const dismiss = useCallback(() => {
    followIntermediatesOperation(null);
    // The Dismiss button unmounts with the panel; keep keyboard focus in the section.
    fallbackFocusRef.current?.focus();
  }, [fallbackFocusRef]);

  return { dismiss, operationId, query };
};
