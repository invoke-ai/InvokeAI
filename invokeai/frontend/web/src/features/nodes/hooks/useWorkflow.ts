import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { buildWorkflow } from 'features/nodes/util/buildWorkflow';
import { useMemo } from 'react';
import { useDebounce } from 'use-debounce';

export const useWorkflow = () => {
  const nodes = useAppSelector((state: RootState) => state.nodes);
  const [debouncedNodes] = useDebounce(nodes, 300);
  const workflow = useMemo(
    () => buildWorkflow(debouncedNodes),
    [debouncedNodes]
  );

  return workflow;
};
