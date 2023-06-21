import { skipToken } from '@reduxjs/toolkit/dist/query';
import { Image, Rect } from 'react-konva';
import { useGetImageDTOQuery } from 'services/apiSlice';
import useImage from 'use-image';
import { CanvasImage } from '../store/canvasTypes';

type IAICanvasImageProps = {
  canvasImage: CanvasImage;
};
const IAICanvasImage = (props: IAICanvasImageProps) => {
  const { width, height, x, y, imageName } = props.canvasImage;
  const { data: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);
  const [image] = useImage(imageDTO?.image_url ?? '', 'anonymous');

  if (!imageDTO) {
    return <Rect x={x} y={y} width={width} height={height} fill="red" />;
  }

  return <Image x={x} y={y} image={image} listening={false} />;
};

export default IAICanvasImage;
