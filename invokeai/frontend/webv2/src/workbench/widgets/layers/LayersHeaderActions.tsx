import { IconButton } from '@workbench/components/ui/Button';
import { MoreHorizontalIcon } from 'lucide-react';

export const LayersHeaderActions = () => (
  <IconButton aria-label="Layer options" color="fg.muted" size="2xs" variant="ghost">
    <MoreHorizontalIcon />
  </IconButton>
);
