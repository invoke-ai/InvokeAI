import { useHandleOldUrls } from 'common/hooks/useHandleOldUrls';
import { useEffect } from 'react';
import { Image } from 'react-konva';
import { ImageDTO } from 'services/api';
import useImage from 'use-image';

type IAICanvasImageProps = {
  image: ImageDTO;
  x: number;
  y: number;
};

const IAICanvasImage = (props: IAICanvasImageProps) => {
  const { image, x, y } = props;
  const handleOldUrls = useHandleOldUrls();
  const [imageElement] = useImage(image.image_url, 'anonymous');

  useEffect(() => {
    if (imageElement) {
      imageElement.onerror = () => handleOldUrls(image);
    }

    return () => {
      if (imageElement) {
        imageElement.onerror = () => undefined;
      }
    };
  }, [imageElement, image, handleOldUrls]);
  return <Image x={x} y={y} image={imageElement} listening={false} />;
};

export default IAICanvasImage;
