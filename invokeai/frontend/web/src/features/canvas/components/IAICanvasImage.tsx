import { useAppDispatch } from 'app/store/storeHooks';
import { useEffect } from 'react';
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
  const [imageElement] = useImage(image.image_url, 'anonymous');
  useEffect(() => {
    if (imageElement) {
      imageElement.onerror = () => {
        dispatch(
          imageUrlsReceived({
            imageName: image.image_name,
            imageOrigin: image.image_origin,
          })
        );
      };
    }
  }, [dispatch, imageElement, image.image_name, image.image_origin]);
  return <Image x={x} y={y} image={imageElement} listening={false} />;
};

export default IAICanvasImage;
