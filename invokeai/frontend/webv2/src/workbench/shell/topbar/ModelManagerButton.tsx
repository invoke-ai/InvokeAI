import { Link } from '@tanstack/react-router';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { BoxIcon } from 'lucide-react';

export const ModelManagerButton = () => (
  <Tooltip content="Model Manager" showArrow>
    <IconButton aria-label="Model Manager" asChild size="sm" variant="ghost">
      <Link to="/models">
        <BoxIcon />
      </Link>
    </IconButton>
  </Tooltip>
);
