import { skipToken } from '@reduxjs/toolkit/query';
import { $authToken } from 'app/store/nanostores/authToken';
import type { CanvasImage } from 'features/canvas/store/canvasTypes';
import { memo } from 'react';
import { Image } from 'react-konva';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import useImage from 'use-image';

import IAICanvasImageErrorFallback from './IAICanvasImageErrorFallback';

type IAICanvasImageProps = {
  canvasImage: CanvasImage;
};
const IAICanvasImage = (props: IAICanvasImageProps) => {
  const { x, y, imageName } = props.canvasImage;
  const { currentData: imageDTO, isError } = useGetImageDTOQuery(imageName ?? skipToken);
  const [image, status] = useImage(imageDTO?.image_url ?? '', $authToken.get() ? 'use-credentials' : 'anonymous');

  if (isError || status === 'failed') {
    return <IAICanvasImageErrorFallback canvasImage={props.canvasImage} />;
  }

  return <Image x={x} y={y} image={image} listening={false} />;
};

export default memo(IAICanvasImage);
