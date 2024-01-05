import { progressAnatomy as parts } from '@chakra-ui/anatomy';
import { createMultiStyleConfigHelpers } from '@chakra-ui/styled-system';
import { generateStripe, getColorVar } from '@chakra-ui/theme-tools';

const { defineMultiStyleConfig, definePartsStyle } =
  createMultiStyleConfigHelpers(parts.keys);

export const progressTheme = defineMultiStyleConfig({
  baseStyle: definePartsStyle(
    ({ theme: t, colorScheme: c, hasStripe, isIndeterminate }) => {
      const bgColor = `${c}.300`;
      const addStripe = !isIndeterminate && hasStripe;
      const gradient = `linear-gradient(
      to right,
      transparent 0%,
      ${getColorVar(t, bgColor)} 50%,
      transparent 100%
    )`;
      return {
        track: {
          borderRadius: '2px',
          bg: 'base.800',
        },
        filledTrack: {
          borderRadius: '2px',
          ...(addStripe && generateStripe()),
          ...(isIndeterminate ? { bgImage: gradient } : { bgColor }),
        },
      };
    }
  ),
});
// export const progressTheme = defineMultiStyleConfig({
//   baseStyle: definePartsStyle(
//     ({ theme: t, colorScheme: c, hasStripe, isIndeterminate }) => {
//       const bgColor = `${c}.500`;
//       const addStripe = !isIndeterminate && hasStripe;
//       const gradient = `linear-gradient(
//       to right,
//       transparent 0%,
//       ${getColorVar(t, bgColor)} 50%,
//       transparent 100%
//     )`;
//       return {
//         track: {
//           borderRadius: '2px',
//           bg: 'base.800',
//         },
//         filledTrack: {
//           borderRadius: '2px',
//           ...(addStripe && generateStripe("xs")),
//           ...(isIndeterminate ? { bgImage: gradient } : { bgColor }),
//         },
//       };
//     }
//   ),
// });
