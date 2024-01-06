import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type { InvSelectOnChange } from 'common/components/InvSelect/types';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterResizeMode } from 'features/controlAdapters/hooks/useControlAdapterResizeMode';
import { controlAdapterResizeModeChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ResizeMode } from 'features/controlAdapters/store/types';
import { isResizeMode } from 'features/controlAdapters/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

const ParamControlAdapterResizeMode = ({ id }: Props) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const resizeMode = useControlAdapterResizeMode(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const options: { label: string; value: ResizeMode }[] = useMemo(
    () => [
      { label: t('controlnet.resize'), value: 'just_resize' },
      { label: t('controlnet.crop'), value: 'crop_resize' },
      { label: t('controlnet.fill'), value: 'fill_resize' },
      { label: t('controlnet.resizeSimple'), value: 'just_resize_simple' },
    ],
    [t]
  );

  const handleResizeModeChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isResizeMode(v?.value)) {
        return;
      }
      dispatch(
        controlAdapterResizeModeChanged({
          id,
          resizeMode: v.value,
        })
      );
    },
    [id, dispatch]
  );

  const value = useMemo(
    () => options.find((o) => o.value === resizeMode),
    [options, resizeMode]
  );

  if (!resizeMode) {
    return null;
  }

  return (
    <InvControl
      label={t('controlnet.resizeMode')}
      feature="controlNetResizeMode"
    >
      <InvSelect
        value={value}
        options={options}
        isDisabled={!isEnabled}
        onChange={handleResizeModeChange}
      />
    </InvControl>
  );
};

export default memo(ParamControlAdapterResizeMode);
