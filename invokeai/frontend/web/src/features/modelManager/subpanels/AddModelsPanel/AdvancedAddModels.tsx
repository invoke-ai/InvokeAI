import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';

import AdvancedAddCheckpoint from './AdvancedAddCheckpoint';
import AdvancedAddDiffusers from './AdvancedAddDiffusers';

export const zManualAddMode = z.enum(['diffusers', 'checkpoint']);
export type ManualAddMode = z.infer<typeof zManualAddMode>;
export const isManualAddMode = (v: unknown): v is ManualAddMode => zManualAddMode.safeParse(v).success;

const AdvancedAddModels = () => {
  const [advancedAddMode, setAdvancedAddMode] = useState<ManualAddMode>('diffusers');

  const { t } = useTranslation();
  const handleChange: ComboboxOnChange = useCallback((v) => {
    if (!isManualAddMode(v?.value)) {
      return;
    }
    setAdvancedAddMode(v.value);
  }, []);

  const options: ComboboxOption[] = useMemo(
    () => [
      { label: t('modelManager.diffusersModels'), value: 'diffusers' },
      { label: t('modelManager.checkpointOrSafetensors'), value: 'checkpoint' },
    ],
    [t]
  );

  const value = useMemo(() => options.find((o) => o.value === advancedAddMode), [options, advancedAddMode]);

  return (
    <Flex flexDirection="column" gap={4} width="100%">
      <FormControl>
        <FormLabel>{t('modelManager.modelType')}</FormLabel>
        <Combobox value={value} options={options} onChange={handleChange} />
      </FormControl>

      <Flex p={4} borderRadius={4} bg="base.850">
        {advancedAddMode === 'diffusers' && <AdvancedAddDiffusers />}
        {advancedAddMode === 'checkpoint' && <AdvancedAddCheckpoint />}
      </Flex>
    </Flex>
  );
};

export default memo(AdvancedAddModels);
