import { useAppDispatch } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterResizeMode } from 'features/controlAdapters/hooks/useControlAdapterResizeMode';
import { controlAdapterResizeModeChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import { ResizeMode } from 'features/controlAdapters/store/types';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

export default function ParamControlAdapterResizeMode({ id }: Props) {
  const isEnabled = useControlAdapterIsEnabled(id);
  const resizeMode = useControlAdapterResizeMode(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const RESIZE_MODE_DATA = [
    { label: t('controlnet.resize'), value: 'just_resize' },
    { label: t('controlnet.crop'), value: 'crop_resize' },
    { label: t('controlnet.fill'), value: 'fill_resize' },
  ];

  const handleResizeModeChange = useCallback(
    (resizeMode: ResizeMode) => {
      dispatch(controlAdapterResizeModeChanged({ id, resizeMode }));
    },
    [id, dispatch]
  );

  if (!resizeMode) {
    return null;
  }

  return (
    <IAIInformationalPopover feature="controlNetResizeMode">
      <IAIMantineSelect
        disabled={!isEnabled}
        label={t('controlnet.resizeMode')}
        data={RESIZE_MODE_DATA}
        value={resizeMode}
        onChange={handleResizeModeChange}
      />
    </IAIInformationalPopover>
  );
}
