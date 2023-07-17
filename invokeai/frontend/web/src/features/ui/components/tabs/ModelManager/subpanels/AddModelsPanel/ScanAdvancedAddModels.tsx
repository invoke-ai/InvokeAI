import { Box, Flex, Text } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { motion } from 'framer-motion';
import { FaTimes } from 'react-icons/fa';
import { setAdvancedAddScanModel } from '../../store/modelManagerSlice';
import AdvancedAddCheckpoint from './AdvancedAddCheckpoint';

export default function ScanAdvancedAddModels() {
  const advancedAddScanModel = useAppSelector(
    (state: RootState) => state.modelmanager.advancedAddScanModel
  );

  const dispatch = useAppDispatch();

  return (
    advancedAddScanModel && (
      <Box
        as={motion.div}
        initial={{ x: -100, opacity: 0 }}
        animate={{ x: 0, opacity: 1, transition: { duration: 0.2 } }}
        sx={{
          display: 'flex',
          flexDirection: 'column',
          minWidth: '50%',
          maxHeight: window.innerHeight - 330,
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
            Add Checkpoint Model
          </Text>
          <IAIIconButton
            icon={<FaTimes />}
            aria-label="Close Advanced"
            onClick={() => dispatch(setAdvancedAddScanModel(null))}
            size="sm"
          />
        </Flex>
        <AdvancedAddCheckpoint
          key={advancedAddScanModel}
          model_path={advancedAddScanModel}
        />
      </Box>
    )
  );
}
