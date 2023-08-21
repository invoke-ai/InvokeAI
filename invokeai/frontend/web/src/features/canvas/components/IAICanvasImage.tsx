import { skipToken } from '@reduxjs/toolkit/dist/query';
import { Image, Rect } from 'react-konva';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import useImage from 'use-image';
import { CanvasImage } from '../store/canvasTypes';
import { $authToken } from 'services/api/client';
import { memo } from 'react';

type IAICanvasImageProps = {
  canvasImage: CanvasImage;
};
const IAICanvasImage = (props: IAICanvasImageProps) => {
  const { width, height, x, y, imageName } = props.canvasImage;
  const { currentData: imageDTO, isError } = useGetImageDTOQuery(
    imageName ?? skipToken
  );
  const [image] = useImage(
    imageDTO?.image_url ?? '',
    $authToken.get() ? 'use-credentials' : 'anonymous'
  );

  if (isError) {
    return <Rect x={x} y={y} width={width} height={height} fill="red" />;
  }

  return <Image x={x} y={y} image={image} listening={false} />;
};

export default memo(IAICanvasImage);
