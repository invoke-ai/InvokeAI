import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowFetchDebounce } from 'features/system/store/configSlice';
import { useGetImageWorkflowQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { useDebounce } from 'use-debounce';

export const useDebouncedImageWorkflow = (imageDTO?: ImageDTO | null) => {
  const workflowFetchDebounce = useAppSelector(selectWorkflowFetchDebounce);

  const [debouncedImageName] = useDebounce(imageDTO?.has_workflow ? imageDTO.image_name : null, workflowFetchDebounce);

  const result = useGetImageWorkflowQuery(debouncedImageName ?? skipToken);

  return result;
};
