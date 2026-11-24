import { chakra } from '@chakra-ui/react';

/** A screen-reader-only comma, inline and zero-size: an absolutely positioned node reads with a space. */
export const OptionNameSeparator = () => <chakra.span fontSize="0">,</chakra.span>;
