import { memo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

import ImageToImageSettings from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageSettings';
import { useAppSelector } from 'app/storeHooks';
import { RootState } from 'app/store';
import { Box } from '@chakra-ui/react';

const AnimatedImageToImagePanel = () => {
  const isImageToImageEnabled = useAppSelector(
    (state: RootState) => state.generation.isImageToImageEnabled
  );

  return (
    <AnimatePresence>
      {isImageToImageEnabled && (
        <motion.div
          initial={{ opacity: 0, scaleX: 0, width: 0 }}
          animate={{ opacity: 1, scaleX: 1, width: '28rem' }}
          exit={{ opacity: 0, scaleX: 0, width: 0 }}
          transition={{ type: 'spring', bounce: 0, duration: 0.35 }}
        >
          <Box sx={{ h: 'full', w: 'full', pl: 4 }}>
            <ImageToImageSettings />
          </Box>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default memo(AnimatedImageToImagePanel);
