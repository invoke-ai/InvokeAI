import type { ComboboxOption, SystemStyleObject } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import type { SingleValue } from 'chakra-react-select';
import { ASPECT_RATIO_OPTIONS } from 'features/parameters/components/ImageSize/constants';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { isAspectRatioID } from 'features/parameters/components/ImageSize/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const AspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();

  const onChange = useCallback(
    (v: SingleValue<ComboboxOption>) => {
      if (!v || !isAspectRatioID(v.value)) {
        return;
      }
      ctx.aspectRatioSelected(v.value);
    },
    [ctx]
  );

  const value = useMemo(
    () => ASPECT_RATIO_OPTIONS.filter((o) => o.value === ctx.aspectRatioState.id)[0],
    [ctx.aspectRatioState.id]
  );

  return (
    <FormControl>
      <FormLabel>{t('parameters.aspect')}</FormLabel>
      <Combobox value={value} onChange={onChange} options={ASPECT_RATIO_OPTIONS} sx={selectStyles} />
    </FormControl>
  );
});

AspectRatioSelect.displayName = 'AspectRatioSelect';

const selectStyles: SystemStyleObject = { minW: 24 };
