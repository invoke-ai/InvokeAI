import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { zChatGPT4oAspectRatioID } from 'features/controlLayers/store/types';
import { aspectRatioChanged, selectAspectRatio, selectModel } from 'features/simpleGeneration/store/slice';
import { isAspectRatio, zAspectRatio } from 'features/simpleGeneration/store/types';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const SimpleTabAspectRatio = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useAppSelector(selectAspectRatio);
  const model = useAppSelector(selectModel);

  const options = useMemo(() => {
    // ChatGPT4o has different aspect ratio options
    if (model === 'chatgpt-4o') {
      return zChatGPT4oAspectRatioID.options;
    }
    // All other models
    return zAspectRatio.options;
  }, [model]);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      if (!isAspectRatio(e.target.value)) {
        return;
      }
      dispatch(aspectRatioChanged({ aspectRatio: e.target.value }));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramAspect">
        <FormLabel m={0}>{t('parameters.aspect')}</FormLabel>
      </InformationalPopover>
      <Select value={id} onChange={onChange}>
        {options.map((ratio) => (
          <option key={ratio} value={ratio}>
            {ratio}
          </option>
        ))}
      </Select>
    </FormControl>
  );
});

SimpleTabAspectRatio.displayName = 'SimpleTabAspectRatio';
