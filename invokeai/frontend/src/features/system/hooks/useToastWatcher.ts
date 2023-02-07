import { useToast } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { toastQueueSelector } from 'features/system/store/systemSelectors';
import { clearToastQueue } from 'features/system/store/systemSlice';
import { useEffect } from 'react';

const useToastWatcher = () => {
  const dispatch = useAppDispatch();
  const toastQueue = useAppSelector(toastQueueSelector);
  const toast = useToast();
  useEffect(() => {
    toastQueue.forEach((t) => {
      toast(t);
    });
    toastQueue.length > 0 && dispatch(clearToastQueue());
  }, [dispatch, toast, toastQueue]);
};

export default useToastWatcher;
