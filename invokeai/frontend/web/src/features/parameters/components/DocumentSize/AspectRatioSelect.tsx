import type { ComboboxOption, SystemStyleObject } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { SingleValue } from 'chakra-react-select';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { documentAspectRatioIdChanged } from 'features/controlLayers/store/canvasV2Slice';
import { ASPECT_RATIO_OPTIONS } from 'features/parameters/components/DocumentSize/constants';
import { isAspectRatioID } from 'features/parameters/components/DocumentSize/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const AspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useAppSelector((s) => s.canvasV2.document.aspectRatio.id);

  const onChange = useCallback(
    (v: SingleValue<ComboboxOption>) => {
      if (!v || !isAspectRatioID(v.value)) {
        return;
      }
      dispatch(documentAspectRatioIdChanged({ id: v.value }));
    },
    [dispatch]
  );

  const value = useMemo(() => ASPECT_RATIO_OPTIONS.filter((o) => o.value === id)[0], [id]);

  return (
    <FormControl>
      <InformationalPopover feature="paramAspect">
        <FormLabel>{t('parameters.aspect')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} onChange={onChange} options={ASPECT_RATIO_OPTIONS} sx={selectStyles} />
    </FormControl>
  );
});

AspectRatioSelect.displayName = 'AspectRatioSelect';

const selectStyles: SystemStyleObject = { minW: 24 };
