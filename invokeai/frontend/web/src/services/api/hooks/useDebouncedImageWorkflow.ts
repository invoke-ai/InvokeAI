import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { useGetImageWorkflowQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { useDebounce } from 'use-debounce';

export const useDebouncedImageWorkflow = (imageDTO?: ImageDTO | null) => {
  const workflowFetchDebounce = useAppSelector((s) => s.config.workflowFetchDebounce ?? 300);

  const [debouncedImageName] = useDebounce(imageDTO?.has_workflow ? imageDTO.image_name : null, workflowFetchDebounce);

  const { data: workflow, isLoading } = useGetImageWorkflowQuery(debouncedImageName ?? skipToken);

  return { workflow, isLoading };
};
