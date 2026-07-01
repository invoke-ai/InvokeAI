import { skipToken } from '@reduxjs/toolkit/query';
import { useGetImageWorkflowQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { useDebounce } from 'use-debounce';

export const useDebouncedImageWorkflow = (imageDTO?: ImageDTO | null) => {
  const [debouncedImageName] = useDebounce(imageDTO?.has_workflow ? imageDTO.image_name : null, 300);

  const result = useGetImageWorkflowQuery(debouncedImageName ?? skipToken);

  return result;
};
