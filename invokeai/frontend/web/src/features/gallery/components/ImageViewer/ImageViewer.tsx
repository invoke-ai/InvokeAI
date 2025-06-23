import { useAppSelector } from 'app/store/storeHooks';
import { selectImageToCompare } from 'features/gallery/components/ImageViewer/common';
import { CurrentImagePreview } from 'features/gallery/components/ImageViewer/CurrentImagePreview';
import { ImageComparison } from 'features/gallery/components/ImageViewer/ImageComparison';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { memo } from 'react';
import { useImageDTO } from 'services/api/endpoints/images';

// type Props = {
//   closeButton?: ReactNode;
// };

// const useFocusRegionOptions = {
//   focusOnMount: true,
// };

// const FOCUS_REGION_STYLES: SystemStyleObject = {
//   display: 'flex',
//   width: 'full',
//   height: 'full',
//   position: 'absolute',
//   flexDirection: 'column',
//   inset: 0,
//   alignItems: 'center',
//   justifyContent: 'center',
//   overflow: 'hidden',
// };

export const ImageViewer = memo(() => {
  const lastSelectedImageName = useAppSelector(selectLastSelectedImage);
  const lastSelectedImageDTO = useImageDTO(lastSelectedImageName);
  const comparisonImageName = useAppSelector(selectImageToCompare);
  const comparisonImageDTO = useImageDTO(comparisonImageName);

  if (lastSelectedImageDTO && comparisonImageDTO) {
    return <ImageComparison firstImage={lastSelectedImageDTO} secondImage={comparisonImageDTO} />;
  }

  return <CurrentImagePreview imageDTO={lastSelectedImageDTO} />;
});

ImageViewer.displayName = 'ImageViewer';
