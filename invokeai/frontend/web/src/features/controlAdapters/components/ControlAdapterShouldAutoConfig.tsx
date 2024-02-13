import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterShouldAutoConfig } from 'features/controlAdapters/hooks/useControlAdapterShouldAutoConfig';
import { controlAdapterAutoConfigToggled } from 'features/controlAdapters/store/controlAdaptersSlice';
import { isNil } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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
    <FormControl isDisabled={!isEnabled}>
      <FormLabel flexGrow={1}>{t('controlnet.autoConfigure')}</FormLabel>
      <Switch isChecked={shouldAutoConfig} onChange={handleShouldAutoConfigChanged} />
    </FormControl>
  );
};

export default memo(ControlAdapterShouldAutoConfig);
