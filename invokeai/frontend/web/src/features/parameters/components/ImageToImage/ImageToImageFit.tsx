import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import type { InvLabelProps } from 'common/components/InvControl/types';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import { setShouldFitToWidthHeight } from 'features/parameters/store/generationSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const labelProps: InvLabelProps = { flexGrow: 1 };

const ImageToImageFit = () => {
  const dispatch = useAppDispatch();

  const shouldFitToWidthHeight = useAppSelector(
    (state: RootState) => state.generation.shouldFitToWidthHeight
  );

  const handleChangeFit = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldFitToWidthHeight(e.target.checked));
    },
    [dispatch]
  );

  const { t } = useTranslation();

  return (
    <InvControl
      label={t('parameters.imageFit')}
      labelProps={labelProps}
      w="full"
    >
      <InvSwitch
        isChecked={shouldFitToWidthHeight}
        onChange={handleChangeFit}
      />
    </InvControl>
  );
};

export default memo(ImageToImageFit);
