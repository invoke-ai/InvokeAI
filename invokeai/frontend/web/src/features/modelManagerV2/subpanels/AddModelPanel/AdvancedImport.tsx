import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, Flex, FormLabel,Text } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';

import { AdvancedImportCheckpoint } from './AdvancedImportCheckpoint';
import { AdvancedImportDiffusers } from './AdvancedImportDiffusers';

export const zManualAddMode = z.enum(['diffusers', 'checkpoint']);
export type ManualAddMode = z.infer<typeof zManualAddMode>;
export const isManualAddMode = (v: unknown): v is ManualAddMode => zManualAddMode.safeParse(v).success;

export const AdvancedImport = () => {
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
    <ScrollableContent>
      <Flex flexDirection="column" gap={4} width="100%" pb={10}>
        <Flex alignItems="flex-end" gap="4">
          <Flex direction="column" gap="3" width="full">
            <FormLabel>{t('modelManager.modelType')}</FormLabel>
            <Combobox value={value} options={options} onChange={handleChange} />
          </Flex>
          <Text px="2" fontSize="xs" textAlign="center">
            {t('modelManager.advancedImportInfo')}
          </Text>
        </Flex>

        <Flex p={4} borderRadius={4} bg="base.850" height="100%">
          {advancedAddMode === 'diffusers' && <AdvancedImportDiffusers />}
          {advancedAddMode === 'checkpoint' && <AdvancedImportCheckpoint />}
        </Flex>
      </Flex>
    </ScrollableContent>
  );
};
