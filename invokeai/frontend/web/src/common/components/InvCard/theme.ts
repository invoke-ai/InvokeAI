import { cardAnatomy } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers } from '@chakra-ui/react';
import { cardVariantLora } from 'features/lora/components/styles';

const { defineMultiStyleConfig } = createMultiStyleConfigHelpers(
  cardAnatomy.keys
);

export const cardTheme = defineMultiStyleConfig({
  variants: { lora: cardVariantLora },
});
