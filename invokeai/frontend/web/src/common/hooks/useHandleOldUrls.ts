import { useAppToaster } from 'app/components/Toaster';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCallback, useState } from 'react';
import { ImageDTO } from 'services/api';
import { imageUrlsReceived } from 'services/thunks/image';

export const useHandleOldUrls = () => {
  const [didGetUrls, setDidGetUrls] = useState(false);
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();
  const shouldFetchImages = useAppSelector(
    (state: RootState) => state.config.shouldFetchImages
  );

  return useCallback(
    (image: ImageDTO | undefined, failedCallback?: () => void) => {
      if (shouldFetchImages && image) {
        if (didGetUrls) {
          toaster({
            title: 'Something went wrong, please refresh',
            status: 'error',
            isClosable: true,
          });
          if (failedCallback) {
            failedCallback();
          }
          return;
        }

        const { image_origin, image_name } = image;

        dispatch(
          imageUrlsReceived({
            imageOrigin: image_origin,
            imageName: image_name,
          })
        );
        setDidGetUrls(true);
        return;
      }

      if (failedCallback) {
        failedCallback();
      }
    },
    [didGetUrls, dispatch, shouldFetchImages, toaster]
  );
};
