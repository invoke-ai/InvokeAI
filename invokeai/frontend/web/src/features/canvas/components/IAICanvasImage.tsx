import { Image } from 'react-konva';
import useImage from 'use-image';

type IAICanvasImageProps = {
  url: string;
  x: number;
  y: number;
};
const IAICanvasImage = (props: IAICanvasImageProps) => {
  const { url, x, y } = props;
  const [image] = useImage(url, 'anonymous');
  return <Image x={x} y={y} image={image} listening={false} />;
};

export default IAICanvasImage;
