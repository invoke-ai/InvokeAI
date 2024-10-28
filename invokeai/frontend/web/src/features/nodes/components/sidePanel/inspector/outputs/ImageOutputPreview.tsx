import { DndImage } from 'features/dnd/DndImage';
import { memo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageOutput } from 'services/api/types';

type Props = {
  output: ImageOutput;
};

const ImageOutputPreview = ({ output }: Props) => {
  const { image } = output;
  const { currentData: imageDTO } = useGetImageDTOQuery(image.image_name);
  if (!imageDTO) {
    return null;
  }

  return <DndImage imageDTO={imageDTO} />;
};

export default memo(ImageOutputPreview);
