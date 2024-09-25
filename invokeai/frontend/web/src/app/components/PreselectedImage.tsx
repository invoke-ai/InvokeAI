import type { UsePreselectedImageArg } from 'features/parameters/hooks/usePreselectedImage';
import { usePreselectedImage } from 'features/parameters/hooks/usePreselectedImage';
import { memo } from 'react';

type Props = {
  selectedImage?: UsePreselectedImageArg;
};

const PreselectedImage = (props: Props) => {
  usePreselectedImage(props.selectedImage);
  return null;
};

export default memo(PreselectedImage);
