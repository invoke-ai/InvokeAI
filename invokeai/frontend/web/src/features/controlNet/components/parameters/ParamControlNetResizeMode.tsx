import { useAppDispatch } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import {
  ControlNetConfig,
  ResizeModes,
  controlNetResizeModeChanged,
} from 'features/controlNet/store/controlNetSlice';
import { ControlNetResizeModePopover } from 'features/informationalPopovers/components/controlNetResizeMode';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type ParamControlNetResizeModeProps = {
  controlNet: ControlNetConfig;
};

export default function ParamControlNetResizeMode(
  props: ParamControlNetResizeModeProps
) {
  const { resizeMode, isEnabled, controlNetId } = props.controlNet;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const RESIZE_MODE_DATA = [
    { label: t('controlnet.resize'), value: 'just_resize' },
    { label: t('controlnet.crop'), value: 'crop_resize' },
    { label: t('controlnet.fill'), value: 'fill_resize' },
  ];

  const handleResizeModeChange = useCallback(
    (resizeMode: ResizeModes) => {
      dispatch(controlNetResizeModeChanged({ controlNetId, resizeMode }));
    },
    [controlNetId, dispatch]
  );

  return (
    <ControlNetResizeModePopover>
      <IAIMantineSelect
        disabled={!isEnabled}
        label={t('controlnet.resizeMode')}
        data={RESIZE_MODE_DATA}
        value={String(resizeMode)}
        onChange={handleResizeModeChange}
      />
    </ControlNetResizeModePopover>
  );
}
