import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectInfillMethod, setInfillMethod } from 'features/controlLayers/store/paramsSlice';
import { zInfillMethod } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetPatchmatchStatusQuery } from 'services/api/endpoints/appInfo';

const ParamInfillMethod = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const infillMethod = useAppSelector(selectInfillMethod);
  const { options } = useGetPatchmatchStatusQuery(undefined, {
    selectFromResult: ({ data: isPatchmatchAvailable }) => {
      if (isPatchmatchAvailable === undefined) {
        // loading...
        return { options: EMPTY_ARRAY };
      }
      if (isPatchmatchAvailable) {
        return { options: zInfillMethod.options.map((o) => ({ label: o, value: o })) };
      }
      return {
        options: zInfillMethod.options.filter((o) => o !== 'patchmatch').map((o) => ({ label: o, value: o })),
      };
    },
  });

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!v || !options.find((o) => o.value === v.value)) {
        return;
      }
      dispatch(setInfillMethod(zInfillMethod.parse(v.value)));
    },
    [dispatch, options]
  );

  const value = useMemo(() => options.find((o) => o.value === infillMethod), [options, infillMethod]);

  return (
    <FormControl isDisabled={options.length === 0}>
      <InformationalPopover feature="infillMethod">
        <FormLabel>{t('parameters.infillMethod')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamInfillMethod);
