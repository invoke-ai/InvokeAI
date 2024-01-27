import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { isSeedBehaviour, seedBehaviourChanged } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsSeedBehaviour = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const seedBehaviour = useAppSelector((s) => s.dynamicPrompts.seedBehaviour);

  const options = useMemo<ComboboxOption[]>(() => {
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

  const handleChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isSeedBehaviour(v?.value)) {
        return;
      }
      dispatch(seedBehaviourChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => options.find((o) => o.value === seedBehaviour), [options, seedBehaviour]);

  return (
    <FormControl>
      <InformationalPopover feature="dynamicPromptsSeedBehaviour" inPortal={false}>
        <FormLabel>{t('dynamicPrompts.seedBehaviour.label')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsSeedBehaviour);
