import { VStack } from '@chakra-ui/react';
import SeedWeights from './SeedWeights';
import VariationAmount from './VariationAmount';

/**
 * Seed & variation options. Includes iteration, seed, seed randomization, variation options.
 */
const VariationsSettings = () => {
  return (
    <VStack gap={2} alignItems="stretch">
      <VariationAmount />
      <SeedWeights />
    </VStack>
  );
};

export default VariationsSettings;
