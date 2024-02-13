import { useMemo } from 'react';

import { ICON_HIGH_CUTOFF, ICON_LOW_CUTOFF } from './constants';

type Dimensions = {
  width: number;
  height: number;
};

type UseAspectRatioPreviewStateArg = {
  width: number;
  height: number;
  containerSize?: Dimensions;
};
type UseAspectRatioPreviewState = (arg: UseAspectRatioPreviewStateArg) => Dimensions & { shouldShowIcon: boolean };

export const useAspectRatioPreviewState: UseAspectRatioPreviewState = ({
  width: _width,
  height: _height,
  containerSize,
}) => {
  const dimensions = useMemo(() => {
    if (!containerSize) {
      return { width: 0, height: 0, shouldShowIcon: false };
    }

    const aspectRatio = _width / _height;
    let width = _width;
    let height = _height;

    if (_width > _height) {
      width = containerSize.width;
      height = width / aspectRatio;
    } else {
      height = containerSize.height;
      width = height * aspectRatio;
    }

    const shouldShowIcon = aspectRatio < ICON_HIGH_CUTOFF && aspectRatio > ICON_LOW_CUTOFF;

    return { width, height, shouldShowIcon };
  }, [_height, _width, containerSize]);

  return dimensions;
};
