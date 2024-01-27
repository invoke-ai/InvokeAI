import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Box, Combobox, Flex, FormControl, FormLabel, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setAdvancedAddScanModel } from 'features/modelManager/store/modelManagerSlice';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

import AdvancedAddCheckpoint from './AdvancedAddCheckpoint';
import AdvancedAddDiffusers from './AdvancedAddDiffusers';
import type { ManualAddMode } from './AdvancedAddModels';
import { isManualAddMode } from './AdvancedAddModels';

const ScanAdvancedAddModels = () => {
  const advancedAddScanModel = useAppSelector((s) => s.modelmanager.advancedAddScanModel);

  const { t } = useTranslation();

  const options: ComboboxOption[] = useMemo(
    () => [
      { label: t('modelManager.diffusersModels'), value: 'diffusers' },
      { label: t('modelManager.checkpointOrSafetensors'), value: 'checkpoint' },
    ],
    [t]
  );

  const [advancedAddMode, setAdvancedAddMode] = useState<ManualAddMode>('diffusers');

  const [isCheckpoint, setIsCheckpoint] = useState<boolean>(true);

  useEffect(() => {
    advancedAddScanModel && ['.ckpt', '.safetensors', '.pth', '.pt'].some((ext) => advancedAddScanModel.endsWith(ext))
      ? setAdvancedAddMode('checkpoint')
      : setAdvancedAddMode('diffusers');
  }, [advancedAddScanModel, setAdvancedAddMode, isCheckpoint]);

  const dispatch = useAppDispatch();

  const handleClickSetAdvanced = useCallback(() => dispatch(setAdvancedAddScanModel(null)), [dispatch]);

  const handleChangeAddMode = useCallback<ComboboxOnChange>((v) => {
    if (!isManualAddMode(v?.value)) {
      return;
    }
    setAdvancedAddMode(v.value);
    if (v.value === 'checkpoint') {
      setIsCheckpoint(true);
    } else {
      setIsCheckpoint(false);
    }
  }, []);

  const value = useMemo(() => options.find((o) => o.value === advancedAddMode), [options, advancedAddMode]);

  if (!advancedAddScanModel) {
    return null;
  }

  return (
    <Box
      display="flex"
      flexDirection="column"
      minWidth="40%"
      maxHeight="calc(100vh - 300px)"
      overflow="scroll"
      p={4}
      gap={4}
      borderRadius={4}
      bg="base.800"
    >
      <Flex justifyContent="space-between" alignItems="center">
        <Text size="xl" fontWeight="semibold">
          {isCheckpoint || advancedAddMode === 'checkpoint' ? 'Add Checkpoint Model' : 'Add Diffusers Model'}
        </Text>
        <IconButton
          icon={<PiXBold />}
          aria-label={t('modelManager.closeAdvanced')}
          onClick={handleClickSetAdvanced}
          size="sm"
        />
      </Flex>
      <FormControl>
        <FormLabel>{t('modelManager.modelType')}</FormLabel>
        <Combobox value={value} options={options} onChange={handleChangeAddMode} />
      </FormControl>
      {isCheckpoint ? (
        <AdvancedAddCheckpoint key={advancedAddScanModel} model_path={advancedAddScanModel} />
      ) : (
        <AdvancedAddDiffusers key={advancedAddScanModel} model_path={advancedAddScanModel} />
      )}
    </Box>
  );
};

export default memo(ScanAdvancedAddModels);
