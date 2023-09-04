import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import {
  ControlNetConfig,
  ResizeModes,
  controlNetResizeModeChanged,
} from 'features/controlNet/store/controlNetSlice';
import { useCallback } from 'react';

type ParamControlNetResizeModeProps = {
  controlNet: ControlNetConfig;
};

const RESIZE_MODE_DATA = [
  { label: 'Resize', value: 'just_resize' },
  { label: 'Crop', value: 'crop_resize' },
  { label: 'Fill', value: 'fill_resize' },
];

export default function ParamControlNetResizeMode(
  props: ParamControlNetResizeModeProps
) {
  const { resizeMode, isEnabled, controlNetId } = props.controlNet;
  const dispatch = useAppDispatch();

  const handleResizeModeChange = useCallback(
    (resizeMode: ResizeModes) => {
      dispatch(controlNetResizeModeChanged({ controlNetId, resizeMode }));
    },
    [controlNetId, dispatch]
  );

  return (
    <IAIMantineSelect
      disabled={!isEnabled}
      label="Resize Mode"
      data={RESIZE_MODE_DATA}
      value={String(resizeMode)}
      onChange={handleResizeModeChange}
    />
  );
}
