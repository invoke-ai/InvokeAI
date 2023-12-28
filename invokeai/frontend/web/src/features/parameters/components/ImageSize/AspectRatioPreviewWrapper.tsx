import { useAppSelector } from 'app/store/storeHooks';
import { AspectRatioPreview } from 'common/components/AspectRatioPreview/AspectRatioPreview';

export const AspectRatioPreviewWrapper = () => {
  const width = useAppSelector((state) => state.generation.width);
  const height = useAppSelector((state) => state.generation.height);

  return <AspectRatioPreview width={width} height={height} />;
};
