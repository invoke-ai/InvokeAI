import { useAppDispatch } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import type { InvLabelProps } from 'common/components/InvControl/types';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterShouldAutoConfig } from 'features/controlAdapters/hooks/useControlAdapterShouldAutoConfig';
import { controlAdapterAutoConfigToggled } from 'features/controlAdapters/store/controlAdaptersSlice';
import { isNil } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

const labelProps: InvLabelProps = {
  flexGrow: 1,
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
    <InvControl
      label={t('controlnet.autoConfigure')}
      isDisabled={!isEnabled}
      labelProps={labelProps}
    >
      <InvSwitch
        isChecked={shouldAutoConfig}
        onChange={handleShouldAutoConfigChanged}
      />
    </InvControl>
  );
};

export default memo(ControlAdapterShouldAutoConfig);
