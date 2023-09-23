import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  SeedBehaviour,
  seedBehaviourChanged,
} from '../store/dynamicPromptsSlice';
import IAIMantineSelectItemWithDescription from 'common/components/IAIMantineSelectItemWithDescription';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';

type Item = {
  label: string;
  value: SeedBehaviour;
  description: string;
};

const ParamDynamicPromptsSeedBehaviour = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const seedBehaviour = useAppSelector(
    (state) => state.dynamicPrompts.seedBehaviour
  );

  const data = useMemo<Item[]>(() => {
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

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }
      dispatch(seedBehaviourChanged(v as SeedBehaviour));
    },
    [dispatch]
  );

  return (
    <IAIInformationalPopover feature="dynamicPromptsSeedBehaviour">
      <IAIMantineSelect
        label={t('dynamicPrompts.seedBehaviour.label')}
        value={seedBehaviour}
        data={data}
        itemComponent={IAIMantineSelectItemWithDescription}
        onChange={handleChange}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamDynamicPromptsSeedBehaviour);
