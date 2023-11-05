import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setHrfEnabled } from 'features/parameters/store/generationSlice';
import { ChangeEvent, useCallback, useEffect } from 'react';
import { roundToMultiple } from 'common/util/roundDownToMultiple';
import {
  setHrfHeight,
  setHrfWidth,
} from 'features/parameters/store/generationSlice';

export const calculateHrfRes = (
  dispatch: ReturnType<typeof useAppDispatch>,
  width: number,
  height: number
) => {
  const aspect = width / height;
  // Default dimension is set to 512
  const dimension = 512;
  const min_dimension = Math.floor(dimension * 0.5);
  const model_area = dimension * dimension; // Assuming square images for model_area

  let init_width;
  let init_height;

  if (aspect > 1.0) {
    init_height = Math.max(min_dimension, Math.sqrt(model_area / aspect));
    init_width = init_height * aspect;
  } else {
    init_width = Math.max(min_dimension, Math.sqrt(model_area * aspect));
    init_height = init_width / aspect;
  }
  // Cap initial height and width to final height and width.
  init_width = Math.min(width, init_width);
  init_height = Math.min(height, init_height);

  const newWidth = roundToMultiple(Math.floor(init_width), 8);
  const newHeight = roundToMultiple(Math.floor(init_height), 8);

  dispatch(setHrfWidth(newWidth));
  dispatch(setHrfHeight(newHeight));
};

export default function ParamHrfToggle() {
  const dispatch = useAppDispatch();
  const tooltip =
    'Generate with a lower initial resolution, upscale to base resolution, process run Image-to-Image.';

  const hrfEnabled = useAppSelector(
    (state: RootState) => state.generation.hrfEnabled
  );
  const width = useAppSelector((state: RootState) => state.generation.width);
  const height = useAppSelector((state: RootState) => state.generation.height);
  const hrfWidth = useAppSelector(
    (state: RootState) => state.generation.hrfWidth
  );
  const hrfHeight = useAppSelector(
    (state: RootState) => state.generation.hrfHeight
  );

  useEffect(() => {
    if (hrfEnabled) {
      calculateHrfRes(dispatch, width, height);
    }
  }, [dispatch, hrfEnabled, width, height]);

  const handleHrfEnabled = useCallback(
    (e: ChangeEvent<HTMLInputElement>) =>
      dispatch(setHrfEnabled(e.target.checked)),
    [dispatch]
  );

  const label = `Enable High Resolution Fix (${hrfWidth}x${hrfHeight})`;

  return (
    <IAISwitch
      label={label}
      isChecked={hrfEnabled}
      onChange={handleHrfEnabled}
      tooltip={tooltip}
    />
  );
}
