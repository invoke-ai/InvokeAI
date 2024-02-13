import { useToast } from '@chakra-ui/react';
import {
  selectSystemSlice,
  setUploadProgressToastId,
} from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';

const UploadProgressToastManager = () => {
  const toast = useToast();
  const dispatch = useDispatch();
  const { uploadProgress, uploadProgressToastId } =
    useSelector(selectSystemSlice);

  useEffect(() => {
    if (uploadProgress) {
      const { progress, processed, total } = uploadProgress;
      const description = `Uploaded ${processed} of ${total} images (${progress.toFixed(
        2
      )}%)`;

      if (!uploadProgressToastId) {
        const newToastId = toast(
          makeToast({
            title: 'Uploading Images',
            description,
            status: 'success',
            duration: null,
          })
        );
        dispatch(setUploadProgressToastId(newToastId));
      } else {
        toast.update(uploadProgressToastId, {
          title: 'Uploading Images',
          description,
          status: 'success',
          duration: null,
        });
      }
    } else if (uploadProgressToastId) {
      setTimeout(() => {
        toast.close(uploadProgressToastId);
        dispatch(setUploadProgressToastId(null));
      }, 2500);
    }
  }, [toast, uploadProgress, uploadProgressToastId, dispatch]);

  return null;
};

export default UploadProgressToastManager;
