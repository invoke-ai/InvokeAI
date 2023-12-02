import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { buildWorkflow } from 'features/nodes/util/workflow/buildWorkflow';
import { useMemo } from 'react';
import { useDebounce } from 'use-debounce';

export const useWorkflow = () => {
  const nodes_ = useAppSelector((state: RootState) => state.nodes.nodes);
  const edges_ = useAppSelector((state: RootState) => state.nodes.edges);
  const workflow_ = useAppSelector((state: RootState) => state.workflow);
  const [nodes] = useDebounce(nodes_, 300);
  const [edges] = useDebounce(edges_, 300);
  const [workflow] = useDebounce(workflow_, 300);
  const builtWorkflow = useMemo(
    () => buildWorkflow({ nodes, edges, workflow }),
    [nodes, edges, workflow]
  );

  return builtWorkflow;
};
