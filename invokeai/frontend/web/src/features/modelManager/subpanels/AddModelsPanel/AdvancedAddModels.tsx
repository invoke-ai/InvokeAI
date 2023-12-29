import { Flex } from '@chakra-ui/react';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';

import AdvancedAddCheckpoint from './AdvancedAddCheckpoint';
import AdvancedAddDiffusers from './AdvancedAddDiffusers';

export const zManualAddMode = z.enum(['diffusers', 'checkpoint']);
export type ManualAddMode = z.infer<typeof zManualAddMode>;
export const isManualAddMode = (v: unknown): v is ManualAddMode =>
  zManualAddMode.safeParse(v).success;

const AdvancedAddModels = () => {
  const [advancedAddMode, setAdvancedAddMode] =
    useState<ManualAddMode>('diffusers');

  const { t } = useTranslation();
  const handleChange: InvSelectOnChange = useCallback((v) => {
    if (!isManualAddMode(v?.value)) {
      return;
    }
    setAdvancedAddMode(v.value);
  }, []);

  const options: InvSelectOption[] = useMemo(
    () => [
      { label: t('modelManager.diffusersModels'), value: 'diffusers' },
      { label: t('modelManager.checkpointOrSafetensors'), value: 'checkpoint' },
    ],
    [t]
  );

  const value = useMemo(
    () => options.find((o) => o.value === advancedAddMode),
    [options, advancedAddMode]
  );

  return (
    <Flex flexDirection="column" gap={4} width="100%">
      <InvControl label={t('modelManager.modelType')}>
        <InvSelect value={value} options={options} onChange={handleChange} />
      </InvControl>

      <Flex p={4} borderRadius={4} bg="base.850">
        {advancedAddMode === 'diffusers' && <AdvancedAddDiffusers />}
        {advancedAddMode === 'checkpoint' && <AdvancedAddCheckpoint />}
      </Flex>
    </Flex>
  );
};

export default memo(AdvancedAddModels);
