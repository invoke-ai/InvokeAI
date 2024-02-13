import { useAppSelector } from 'app/store/storeHooks';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import { calculateNewSize } from 'features/parameters/components/ImageSize/calculateNewSize';
import { ASPECT_RATIO_MAP, initialAspectRatioState } from 'features/parameters/components/ImageSize/constants';
import type { AspectRatioID, AspectRatioState } from 'features/parameters/components/ImageSize/types';
import { selectOptimalDimension } from 'features/parameters/store/generationSlice';
import { createContext, useCallback, useContext, useMemo } from 'react';

export type ImageSizeContextInnerValue = {
  width: number;
  height: number;
  aspectRatioState: AspectRatioState;
  onChangeWidth: (width: number) => void;
  onChangeHeight: (height: number) => void;
  onChangeAspectRatioState: (aspectRatioState: AspectRatioState) => void;
};

export type ImageSizeContext = {
  width: number;
  height: number;
  aspectRatioState: AspectRatioState;
  aspectRatioSelected: (aspectRatioID: AspectRatioID) => void;
  dimensionsSwapped: () => void;
  widthChanged: (width: number) => void;
  heightChanged: (height: number) => void;
  isLockedToggled: () => void;
  setOptimalSize: () => void;
};

export const ImageSizeContext = createContext<ImageSizeContextInnerValue | null>(null);

export const useImageSizeContext = (): ImageSizeContext => {
  const _ctx = useContext(ImageSizeContext);
  const optimalDimension = useAppSelector(selectOptimalDimension);

  if (!_ctx) {
    throw new Error('useImageSizeContext must be used within a ImageSizeContext.Provider');
  }

  const aspectRatioSelected = useCallback(
    (aspectRatioID: AspectRatioID) => {
      const state: AspectRatioState = {
        ..._ctx.aspectRatioState,
        id: aspectRatioID,
      };
      if (state.id === 'Free') {
        // If the new aspect ratio is free, we only unlock
        state.isLocked = false;
      } else {
        // The new aspect ratio not free, so we need to coerce the size & lock
        state.isLocked = true;
        state.value = ASPECT_RATIO_MAP[state.id].ratio;
        const { width, height } = calculateNewSize(state.value, _ctx.width * _ctx.height);
        _ctx.onChangeWidth(width);
        _ctx.onChangeHeight(height);
      }
      _ctx.onChangeAspectRatioState(state);
    },
    [_ctx]
  );
  const dimensionsSwapped = useCallback(() => {
    const state = {
      ..._ctx.aspectRatioState,
    };
    // We always invert the aspect ratio
    state.value = 1 / state.value;
    if (state.id === 'Free') {
      // If the aspect ratio is free, we just swap the dimensions
      const newWidth = _ctx.height;
      const newHeight = _ctx.width;
      _ctx.onChangeWidth(newWidth);
      _ctx.onChangeHeight(newHeight);
    } else {
      // Else we need to calculate the new size
      const { width, height } = calculateNewSize(state.value, _ctx.width * _ctx.height);
      _ctx.onChangeWidth(width);
      _ctx.onChangeHeight(height);
      // Update the aspect ratio ID to match the new aspect ratio
      state.id = ASPECT_RATIO_MAP[state.id].inverseID;
    }
    _ctx.onChangeAspectRatioState(state);
  }, [_ctx]);

  const widthChanged = useCallback(
    (width: number) => {
      let height = _ctx.height;
      const state = { ..._ctx.aspectRatioState };
      if (state.isLocked) {
        // When locked, we calculate the new height based on the aspect ratio
        height = roundToMultiple(width / state.value, 8);
      } else {
        // Else we unlock, set the aspect ratio to free, and update the aspect ratio itself
        state.isLocked = false;
        state.id = 'Free';
        state.value = width / height;
      }
      _ctx.onChangeWidth(width);
      _ctx.onChangeHeight(height);
      _ctx.onChangeAspectRatioState(state);
    },
    [_ctx]
  );

  const heightChanged = useCallback(
    (height: number) => {
      let width = _ctx.width;
      const state = { ..._ctx.aspectRatioState };
      if (state.isLocked) {
        // When locked, we calculate the new width based on the aspect ratio
        width = roundToMultiple(height * state.value, 8);
      } else {
        // Else we unlock, set the aspect ratio to free, and update the aspect ratio itself
        state.isLocked = false;
        state.id = 'Free';
        state.value = width / height;
      }
      _ctx.onChangeWidth(width);
      _ctx.onChangeHeight(height);
      _ctx.onChangeAspectRatioState(state);
    },
    [_ctx]
  );

  const isLockedToggled = useCallback(() => {
    const state = { ..._ctx.aspectRatioState };
    state.isLocked = !state.isLocked;
    _ctx.onChangeAspectRatioState(state);
  }, [_ctx]);

  const setOptimalSize = useCallback(() => {
    if (_ctx.aspectRatioState.isLocked) {
      const { width, height } = calculateNewSize(_ctx.aspectRatioState.value, optimalDimension * optimalDimension);
      _ctx.onChangeWidth(width);
      _ctx.onChangeHeight(height);
    } else {
      _ctx.onChangeAspectRatioState({ ...initialAspectRatioState });
      _ctx.onChangeWidth(optimalDimension);
      _ctx.onChangeHeight(optimalDimension);
    }
  }, [_ctx, optimalDimension]);

  const ctx = useMemo(
    () => ({
      ..._ctx,
      aspectRatioSelected,
      dimensionsSwapped,
      widthChanged,
      heightChanged,
      isLockedToggled,
      setOptimalSize,
    }),
    [_ctx, aspectRatioSelected, dimensionsSwapped, heightChanged, isLockedToggled, setOptimalSize, widthChanged]
  );

  return ctx;
};
