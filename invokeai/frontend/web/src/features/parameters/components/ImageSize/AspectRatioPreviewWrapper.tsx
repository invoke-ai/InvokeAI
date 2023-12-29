import { useAppSelector } from 'app/store/storeHooks';
import { AspectRatioPreview } from 'common/components/AspectRatioPreview/AspectRatioPreview';
import { memo } from 'react';

export const AspectRatioPreviewWrapper = memo(() => {
  const width = useAppSelector((state) => state.generation.width);
  const height = useAppSelector((state) => state.generation.height);

  return <AspectRatioPreview width={width} height={height} />;
});

AspectRatioPreviewWrapper.displayName = 'AspectRatioPreviewWrapper';
