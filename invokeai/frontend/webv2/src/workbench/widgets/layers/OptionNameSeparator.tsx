import { chakra } from '@chakra-ui/react';

/**
 * A comma only assistive technology reads, between an option's name and its hint ("Depth, 2 models"). Inline and
 * zero-size rather than visually hidden: an absolutely positioned node is read with a space before it.
 */
export const OptionNameSeparator = () => <chakra.span fontSize="0">,</chakra.span>;
