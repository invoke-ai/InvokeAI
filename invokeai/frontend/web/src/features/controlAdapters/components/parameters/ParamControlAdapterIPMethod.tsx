import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useControlAdapterIPMethod } from 'features/controlAdapters/hooks/useControlAdapterIPMethod';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { controlAdapterIPMethodChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { IPMethod } from 'features/controlAdapters/store/types';
import { isIPMethod } from 'features/controlAdapters/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

const ParamControlAdapterIPMethod = ({ id }: Props) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const method = useControlAdapterIPMethod(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const options: { label: string; value: IPMethod }[] = useMemo(
    () => [
      { label: t('controlnet.full'), value: 'full' },
      { label: `${t('controlnet.style')} (${t('common.beta')})`, value: 'style' },
      { label: `${t('controlnet.composition')} (${t('common.beta')})`, value: 'composition' },
    ],
    [t]
  );

  const handleIPMethodChanged = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isIPMethod(v?.value)) {
        return;
      }
      dispatch(
        controlAdapterIPMethodChanged({
          id,
          method: v.value,
        })
      );
    },
    [id, dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === method), [options, method]);

  if (!method) {
    return null;
  }

  return (
    <FormControl>
      <InformationalPopover feature="controlNetResizeMode">
        <FormLabel>{t('controlnet.ipAdapterMethod')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} isDisabled={!isEnabled} onChange={handleIPMethodChanged} />
    </FormControl>
  );
};

export default memo(ParamControlAdapterIPMethod);
