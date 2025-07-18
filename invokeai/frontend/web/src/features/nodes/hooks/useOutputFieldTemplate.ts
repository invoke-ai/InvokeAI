import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import type { FieldOutputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';

export const useOutputFieldTemplate = (fieldName: string): FieldOutputTemplate => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(() => ctx.buildSelectOutputFieldTemplateOrThrow(fieldName), [ctx, fieldName]);
  return useAppSelector(selector);
};
