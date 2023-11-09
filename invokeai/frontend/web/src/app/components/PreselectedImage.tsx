import { usePreselectedImage } from 'features/parameters/hooks/usePreselectedImage';
import { memo } from 'react';

type Props = {
  selectedImage?: {
    imageName: string;
    action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
  };
};

const PreselectedImage = (props: Props) => {
  usePreselectedImage(props.selectedImage);
  return null;
};

export default memo(PreselectedImage);
