import { Icon } from '@chakra-ui/react';
import { Link } from '@tanstack/react-router';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { BlocksIcon } from 'lucide-react';

export const NodesManagerButton = () => (
  <Tooltip content="Nodes Manager" showArrow>
    <IconButton aria-label="Nodes Manager" asChild size="sm" variant="ghost">
      <Link to="/nodes">
        <Icon as={BlocksIcon} />
      </Link>
    </IconButton>
  </Tooltip>
);
