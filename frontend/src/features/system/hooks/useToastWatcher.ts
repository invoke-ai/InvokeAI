import { useToast } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store';
import { useEffect } from 'react';
import { toastQueueSelector } from 'features/system/store/systemSelectors';
import { clearToastQueue } from 'features/system/store/systemSlice';

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
