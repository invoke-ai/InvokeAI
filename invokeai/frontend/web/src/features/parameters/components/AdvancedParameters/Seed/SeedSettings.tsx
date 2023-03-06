import { VStack } from '@chakra-ui/react';
import Perlin from './Perlin';
import RandomizeSeed from './RandomizeSeed';
import Seed from './Seed';
import Threshold from './Threshold';

/**
 * Seed & variation options. Includes iteration, seed, seed randomization, variation options.
 */
const SeedSettings = () => {
  return (
    <VStack gap={2} alignItems="stretch">
      <RandomizeSeed />
      <Seed />
      <Threshold />
      <Perlin />
    </VStack>
  );
};

export default SeedSettings;
