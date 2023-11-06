import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useDebounce } from 'use-debounce';
import { useGetWorkflowQuery } from '../endpoints/workflows';

export const useDebouncedWorkflow = (workflowId?: string | null) => {
  const workflowFetchDebounce = useAppSelector(
    (state) => state.config.workflowFetchDebounce
  );

  const [debouncedWorkflowID] = useDebounce(
    workflowId,
    workflowFetchDebounce ?? 0
  );

  const { data: workflow, isLoading } = useGetWorkflowQuery(
    debouncedWorkflowID ?? skipToken
  );

  return { workflow, isLoading };
};
