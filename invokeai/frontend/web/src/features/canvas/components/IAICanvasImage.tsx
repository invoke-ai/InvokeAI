import { useAppToaster } from 'app/components/Toaster';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEffect, useState } from 'react';
import { Image } from 'react-konva';
import { ImageDTO } from 'services/api';
import { imageUrlsReceived } from 'services/thunks/image';
import useImage from 'use-image';

type IAICanvasImageProps = {
  image: ImageDTO;
  x: number;
  y: number;
};

const IAICanvasImage = (props: IAICanvasImageProps) => {
  const { image, x, y } = props;
  const dispatch = useAppDispatch();
  const toaster = useAppToaster();
  const shouldFetchImages = useAppSelector(
    (state: RootState) => state.config.shouldFetchImages
  );
  const [didGetUrls, setDidGetUrls] = useState(false);
  const [imageElement] = useImage(image.image_url, 'anonymous');

  useEffect(() => {
    if (imageElement) {
      imageElement.onerror = () => {
        if (shouldFetchImages && image) {
          if (didGetUrls) {
            toaster({
              title: 'Something went wrong, please refresh',
              status: 'error',
              isClosable: true,
            });
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
        }
      };
    }
  }, [dispatch, imageElement, image, shouldFetchImages, didGetUrls, toaster]);
  return <Image x={x} y={y} image={imageElement} listening={false} />;
};

export default IAICanvasImage;
