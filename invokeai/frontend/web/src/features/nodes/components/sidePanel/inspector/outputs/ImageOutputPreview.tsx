import IAIDndImage from 'common/components/IAIDndImage';
import { memo } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { ImageOutput } from 'services/api/types';

type Props = {
  output: ImageOutput;
};

const ImageOutputPreview = ({ output }: Props) => {
  const { image } = output;
  const { data: imageDTO } = useGetImageDTOQuery(image.image_name);

  return <IAIDndImage imageDTO={imageDTO} />;
};

export default memo(ImageOutputPreview);
