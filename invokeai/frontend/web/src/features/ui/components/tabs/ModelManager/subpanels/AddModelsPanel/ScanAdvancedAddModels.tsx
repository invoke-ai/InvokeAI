import { Box, Flex, Text } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { FaTimes } from 'react-icons/fa';
import { setAdvancedAddScanModel } from '../../store/modelManagerSlice';
import AdvancedAddCheckpoint from './AdvancedAddCheckpoint';
import AdvancedAddDiffusers from './AdvancedAddDiffusers';
import { ManualAddMode, advancedAddModeData } from './AdvancedAddModels';

export default function ScanAdvancedAddModels() {
  const advancedAddScanModel = useAppSelector(
    (state: RootState) => state.modelmanager.advancedAddScanModel
  );

  const [advancedAddMode, setAdvancedAddMode] =
    useState<ManualAddMode>('diffusers');

  const [isCheckpoint, setIsCheckpoint] = useState<boolean>(true);

  useEffect(() => {
    advancedAddScanModel &&
    ['.ckpt', '.safetensors', '.pth', '.pt'].some((ext) =>
      advancedAddScanModel.endsWith(ext)
    )
      ? setAdvancedAddMode('checkpoint')
      : setAdvancedAddMode('diffusers');
  }, [advancedAddScanModel, setAdvancedAddMode, isCheckpoint]);

  const dispatch = useAppDispatch();

  if (!advancedAddScanModel) {
    return null;
  }

  return (
    <Box
      as={motion.div}
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1, transition: { duration: 0.2 } }}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        minWidth: '40%',
        maxHeight: window.innerHeight - 300,
        overflow: 'scroll',
        p: 4,
        gap: 4,
        borderRadius: 4,
        bg: 'base.200',
        _dark: {
          bg: 'base.800',
        },
      }}
    >
      <Flex justifyContent="space-between" alignItems="center">
        <Text size="xl" fontWeight={600}>
          {isCheckpoint || advancedAddMode === 'checkpoint'
            ? 'Add Checkpoint Model'
            : 'Add Diffusers Model'}
        </Text>
        <IAIIconButton
          icon={<FaTimes />}
          aria-label="Close Advanced"
          onClick={() => dispatch(setAdvancedAddScanModel(null))}
          size="sm"
        />
      </Flex>
      <IAIMantineSelect
        label="Model Type"
        value={advancedAddMode}
        data={advancedAddModeData}
        onChange={(v) => {
          if (!v) {
            return;
          }
          setAdvancedAddMode(v as ManualAddMode);
          if (v === 'checkpoint') {
            setIsCheckpoint(true);
          } else {
            setIsCheckpoint(false);
          }
        }}
      />
      {isCheckpoint ? (
        <AdvancedAddCheckpoint
          key={advancedAddScanModel}
          model_path={advancedAddScanModel}
        />
      ) : (
        <AdvancedAddDiffusers
          key={advancedAddScanModel}
          model_path={advancedAddScanModel}
        />
      )}
    </Box>
  );
}
