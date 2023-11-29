import { useAppDispatch } from 'app/store/storeHooks';
import { workflowLoadRequested } from 'features/nodes/store/actions';
import { useCallback } from 'react';
import { useLazyGetImageWorkflowQuery } from 'services/api/endpoints/images';

export const useGetAndLoadEmbeddedWorkflow = (
  image_name: string | undefined
) => {
  const dispatch = useAppDispatch();
  const [_trigger, result] = useLazyGetImageWorkflowQuery();
  const trigger = useCallback(() => {
    if (!image_name) {
      return;
    }
    _trigger(image_name).then((workflow) => {
      dispatch(workflowLoadRequested(workflow.data));
    });
  }, [dispatch, _trigger, image_name]);

  return [trigger, result];
};
