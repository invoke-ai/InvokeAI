import { useAppSelector } from 'app/store/storeHooks';
import { buildSelectElement } from 'features/nodes/store/selectors';
import type { FormElement } from 'features/nodes/types/workflow';
import { useMemo } from 'react';

export const useElement = (id: string): FormElement | undefined => {
  const selector = useMemo(() => buildSelectElement(id), [id]);
  const element = useAppSelector(selector);
  return element;
};
