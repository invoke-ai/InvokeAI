import { IconButton } from '@chakra-ui/react';
import { PiDotsThreeBold } from 'react-icons/pi';

export const LayersHeaderActions = () => (
  <IconButton aria-label="Layer options" color="fg.muted" size="2xs" variant="ghost">
    <PiDotsThreeBold />
  </IconButton>
);
