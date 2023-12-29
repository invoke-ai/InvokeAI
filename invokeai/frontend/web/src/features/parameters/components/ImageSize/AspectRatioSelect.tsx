import type { SystemStyleObject } from '@chakra-ui/styled-system';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { SingleValue } from 'chakra-react-select';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type { InvSelectOption } from 'common/components/InvSelect/types';
import { ASPECT_RATIO_OPTIONS } from 'features/parameters/components/ImageSize/constants';
import { isAspectRatioID } from 'features/parameters/components/ImageSize/types';
import { aspectRatioSelected } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { LockAspectRatioButton } from './LockAspectRatioButton';
import { SetOptimalSizeButton } from './SetOptimalSizeButton';
import { SwapDimensionsButton } from './SwapDimensionsButton';

export const AspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const aspectRatioID = useAppSelector(
    (state) => state.generation.aspectRatio.id
  );

  const onChange = useCallback(
    (v: SingleValue<InvSelectOption>) => {
      if (!v || !isAspectRatioID(v.value)) {
        return;
      }
      dispatch(aspectRatioSelected(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => ASPECT_RATIO_OPTIONS.filter((o) => o.value === aspectRatioID)[0],
    [aspectRatioID]
  );

  return (
    <InvControl label={t('parameters.aspect')}>
      <InvSelect
        value={value}
        onChange={onChange}
        options={ASPECT_RATIO_OPTIONS}
        sx={selectStyles}
      />
      <SwapDimensionsButton />
      <LockAspectRatioButton />
      <SetOptimalSizeButton />
    </InvControl>
  );
});

AspectRatioSelect.displayName = 'AspectRatioSelect';

const selectStyles: SystemStyleObject = { minW: 48 };
