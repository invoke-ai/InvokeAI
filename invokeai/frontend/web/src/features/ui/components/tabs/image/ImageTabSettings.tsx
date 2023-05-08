import { memo } from 'react';
import OverlayScrollable from '../../common/OverlayScrollable';
import ImageToImageSettings from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageSettings';
import {
  Box,
  ButtonGroup,
  Collapse,
  Flex,
  Heading,
  HStack,
  Image,
  Spacer,
  useDisclosure,
  VStack,
} from '@chakra-ui/react';
import { motion } from 'framer-motion';

import IAIButton from 'common/components/IAIButton';
import ImageFit from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageFit';
import ImageToImageStrength from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageStrength';
import IAIIconButton from 'common/components/IAIIconButton';

import { useTranslation } from 'react-i18next';
import { useState } from 'react';
import { FaUndo, FaUpload } from 'react-icons/fa';
import ImagePromptHeading from 'common/components/ImageToImageSettingsHeader';
import InitialImagePreview from 'features/parameters/components/AdvancedParameters/ImageToImage/InitialImagePreview';

const CreateImageSettings = () => {
  return (
    <OverlayScrollable>
      <Flex
        sx={{
          gap: 2,
          flexDirection: 'column',
          h: 'full',
          w: 'full',
          position: 'absolute',
          borderRadius: 'base',
          // bg: 'base.850',
          // p: 2,
        }}
      >
        <ImagePromptHeading />
        <InitialImagePreview />
        <ImageToImageStrength />
        <ImageFit />
      </Flex>
    </OverlayScrollable>
  );
};

export default memo(CreateImageSettings);
