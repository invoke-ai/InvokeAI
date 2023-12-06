import { useAppSelector } from 'app/store/storeHooks';
import { buildWorkflow } from 'features/nodes/util/workflow/buildWorkflow';
import { omit } from 'lodash-es';
import { useMemo } from 'react';
import { useDebounce } from 'use-debounce';

export const useWorkflow = () => {
  const nodes_ = useAppSelector((state) => state.nodes.nodes);
  const edges_ = useAppSelector((state) => state.nodes.edges);
  const workflow_ = useAppSelector((state) => state.workflow);
  const [nodes] = useDebounce(nodes_, 300);
  const [edges] = useDebounce(edges_, 300);
  const [workflow] = useDebounce(workflow_, 300);
  const builtWorkflow = useMemo(
    () =>
      buildWorkflow({ nodes, edges, workflow: omit(workflow, 'isTouched') }),
    [nodes, edges, workflow]
  );

  return builtWorkflow;
};
