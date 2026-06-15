type GetMasonryRenderStateArg = {
  hasMeasuredColumnCount: boolean;
  imageCount: number;
  isLoading: boolean;
};

export type MasonryRenderState = 'empty' | 'loading' | 'measuring' | 'ready';

export const getMasonryRenderState = ({
  hasMeasuredColumnCount,
  imageCount,
  isLoading,
}: GetMasonryRenderStateArg): MasonryRenderState => {
  if (isLoading) {
    return 'loading';
  }

  if (imageCount === 0) {
    return 'empty';
  }

  if (!hasMeasuredColumnCount) {
    return 'measuring';
  }

  return 'ready';
};
