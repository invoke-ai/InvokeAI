import type { SystemStyleObject } from '@chakra-ui/styled-system';
import type { SingleValue } from 'chakra-react-select';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type { InvSelectOption } from 'common/components/InvSelect/types';
import { ASPECT_RATIO_OPTIONS } from 'features/parameters/components/ImageSize/constants';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { isAspectRatioID } from 'features/parameters/components/ImageSize/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const AspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();

  const onChange = useCallback(
    (v: SingleValue<InvSelectOption>) => {
      if (!v || !isAspectRatioID(v.value)) {
        return;
      }
      ctx.aspectRatioSelected(v.value);
    },
    [ctx]
  );

  const value = useMemo(
    () =>
      ASPECT_RATIO_OPTIONS.filter(
        (o) => o.value === ctx.aspectRatioState.id
      )[0],
    [ctx.aspectRatioState.id]
  );

  return (
    <InvControl label={t('parameters.aspect')}>
      <InvSelect
        value={value}
        onChange={onChange}
        options={ASPECT_RATIO_OPTIONS}
        sx={selectStyles}
      />
    </InvControl>
  );
});

AspectRatioSelect.displayName = 'AspectRatioSelect';

const selectStyles: SystemStyleObject = { minW: 24 };
