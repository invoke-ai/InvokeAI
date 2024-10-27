import { DndImage } from 'features/dnd2/DndImage';
import { memo, useId } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageOutput } from 'services/api/types';

type Props = {
  output: ImageOutput;
};

const ImageOutputPreview = ({ output }: Props) => {
  const { image } = output;
  const { currentData: imageDTO } = useGetImageDTOQuery(image.image_name);
  const dndId = useId();
  if (!imageDTO) {
    return null;
  }

  return <DndImage dndId={dndId} imageDTO={imageDTO} />;
};

export default memo(ImageOutputPreview);
