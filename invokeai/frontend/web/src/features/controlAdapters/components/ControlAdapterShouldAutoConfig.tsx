import { useAppDispatch } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { controlAdapterAutoConfigToggled } from 'features/controlAdapters/store/controlAdaptersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterShouldAutoConfig } from 'features/controlAdapters/hooks/useControlAdapterShouldAutoConfig';
import { isNil } from 'lodash-es';

type Props = {
  id: string;
};

const ControlAdapterShouldAutoConfig = ({ id }: Props) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const shouldAutoConfig = useControlAdapterShouldAutoConfig(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleShouldAutoConfigChanged = useCallback(() => {
    dispatch(controlAdapterAutoConfigToggled({ id }));
  }, [id, dispatch]);

  if (isNil(shouldAutoConfig)) {
    return null;
  }

  return (
    <IAISwitch
      label={t('controlnet.autoConfigure')}
      aria-label={t('controlnet.autoConfigure')}
      isChecked={shouldAutoConfig}
      onChange={handleShouldAutoConfigChanged}
      isDisabled={!isEnabled}
    />
  );
};

export default memo(ControlAdapterShouldAutoConfig);
