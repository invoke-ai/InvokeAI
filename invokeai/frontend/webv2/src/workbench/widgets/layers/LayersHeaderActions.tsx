import { MoreHorizontalIcon } from 'lucide-react';

import { IconButton } from '../../components/ui/Button';

export const LayersHeaderActions = () => (
  <IconButton aria-label="Layer options" color="fg.muted" size="2xs" variant="ghost">
    <MoreHorizontalIcon />
  </IconButton>
);
