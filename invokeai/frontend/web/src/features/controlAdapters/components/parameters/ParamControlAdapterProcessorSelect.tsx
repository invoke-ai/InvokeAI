import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterProcessorNode } from 'features/controlAdapters/hooks/useControlAdapterProcessorNode';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { controlAdapterProcessortTypeChanged } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlAdapterProcessorType } from 'features/controlAdapters/store/types';
import { configSelector } from 'features/system/store/configSelectors';
import { map } from 'lodash-es';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

const selectOptions = createMemoizedSelector(configSelector, (config) => {
  const options: InvSelectOption[] = map(CONTROLNET_PROCESSORS, (p) => ({
    value: p.type,
    label: p.label,
  }))
    .sort((a, b) =>
      // sort 'none' to the top
      a.value === 'none'
        ? -1
        : b.value === 'none'
          ? 1
          : a.label.localeCompare(b.label)
    )
    .filter(
      (d) =>
        !config.sd.disabledControlNetProcessors.includes(
          d.value as ControlAdapterProcessorType
        )
    );

  return options;
});

const ParamControlAdapterProcessorSelect = ({ id }: Props) => {
  const isEnabled = useControlAdapterIsEnabled(id);
  const processorNode = useControlAdapterProcessorNode(id);
  const dispatch = useAppDispatch();
  const options = useAppSelector(selectOptions);
  const { t } = useTranslation();

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!v) {
        return;
      }
      dispatch(
        controlAdapterProcessortTypeChanged({
          id,
          processorType: v.value as ControlAdapterProcessorType, // TODO: need runtime check...
        })
      );
    },
    [id, dispatch]
  );
  const value = useMemo(
    () => options.find((o) => o.value === processorNode?.type),
    [options, processorNode]
  );

  if (!processorNode) {
    return null;
  }
  return (
    <InvControl label={t('controlnet.processor')} isDisabled={!isEnabled}>
      <InvSelect value={value} options={options} onChange={onChange} />
    </InvControl>
  );
};

export default memo(ParamControlAdapterProcessorSelect);
