import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import {
  isSeedBehaviour,
  seedBehaviourChanged,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsSeedBehaviour = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const seedBehaviour = useAppSelector(
    (s) => s.dynamicPrompts.seedBehaviour
  );

  const options = useMemo<InvSelectOption[]>(() => {
    return [
      {
        value: 'PER_ITERATION',
        label: t('dynamicPrompts.seedBehaviour.perIterationLabel'),
        description: t('dynamicPrompts.seedBehaviour.perIterationDesc'),
      },
      {
        value: 'PER_PROMPT',
        label: t('dynamicPrompts.seedBehaviour.perPromptLabel'),
        description: t('dynamicPrompts.seedBehaviour.perPromptDesc'),
      },
    ];
  }, [t]);

  const handleChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isSeedBehaviour(v?.value)) {
        return;
      }
      dispatch(seedBehaviourChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => options.find((o) => o.value === seedBehaviour),
    [options, seedBehaviour]
  );

  return (
    <InvControl
      label={t('dynamicPrompts.seedBehaviour.label')}
      feature="dynamicPromptsSeedBehaviour"
      renderInfoPopoverInPortal={false}
    >
      <InvSelect value={value} options={options} onChange={handleChange} />
    </InvControl>
  );
};

export default memo(ParamDynamicPromptsSeedBehaviour);
