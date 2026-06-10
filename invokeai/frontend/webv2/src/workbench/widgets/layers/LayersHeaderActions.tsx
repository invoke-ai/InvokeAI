import { PiDotsThreeBold } from 'react-icons/pi';

import { IconButton } from '../../components/ui/Button';

export const LayersHeaderActions = () => (
  <IconButton aria-label="Layer options" color="fg.muted" size="2xs" variant="ghost">
    <PiDotsThreeBold />
  </IconButton>
);
