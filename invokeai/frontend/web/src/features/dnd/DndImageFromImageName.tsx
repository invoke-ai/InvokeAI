import { IAINoContentFallback, IAINoContentFallbackWithSpinner } from 'common/components/IAIImageFallback';
import { DndImage } from 'features/dnd/DndImage';
import { memo } from 'react';
import { PiExclamationMarkBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

/* eslint-disable-next-line @typescript-eslint/no-namespace */
namespace DndImageFromImageName {
  export interface Props extends Omit<DndImage.Props, 'imageDTO'> {
    imageName: string;
  }
}

export const DndImageFromImageName = memo(({ imageName, ...rest }: DndImageFromImageName.Props) => {
  const query = useGetImageDTOQuery(imageName);
  if (query.isLoading) {
    return <IAINoContentFallbackWithSpinner />;
  }
  if (!query.data) {
    return <IAINoContentFallback icon={<PiExclamationMarkBold />} />;
  }

  return <DndImage imageDTO={query.data} {...rest} />;
});

DndImageFromImageName.displayName = 'DndImageFromImageName';
