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
import ImageToImageFit from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageFit';
import ImageToImageStrength from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageStrength';
import IAIIconButton from 'common/components/IAIIconButton';

import { useTranslation } from 'react-i18next';
import InitialImagePreview from './InitialImagePreview';
import { useState } from 'react';
import { FaUndo, FaUpload } from 'react-icons/fa';
import InitialImageButtons from 'common/components/ImageToImageSettingsHeader';

export default function ImageToImageSettings() {
  const { t } = useTranslation();
  return (
    <VStack gap={2} w="full" alignItems="stretch">
      <InitialImageButtons />
      <InitialImagePreview />
      <ImageToImageStrength />
      <ImageToImageFit />
    </VStack>
  );
}
