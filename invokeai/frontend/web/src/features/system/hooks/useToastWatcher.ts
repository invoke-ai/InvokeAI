import { useToast, UseToastOptions } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { toastQueueSelector } from 'features/system/store/systemSelectors';
import { clearToastQueue } from 'features/system/store/systemSlice';
import { useEffect } from 'react';

export type MakeToastArg = string | UseToastOptions;

export const makeToast = (arg: MakeToastArg): UseToastOptions => {
  if (typeof arg === 'string') {
    return {
      title: arg,
      status: 'info',
      isClosable: true,
      duration: 2500,
    };
  }

  return { status: 'info', isClosable: true, duration: 2500, ...arg };
};

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
