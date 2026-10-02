import { Icon } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { RotateCcwIcon } from 'lucide-react';
import { useCallback } from 'react';

/** Render only for modified values and supply a specific reset label. */
export const ModelDefaultButton = ({ label, onClick }: { label: string; onClick: () => void }) => {
  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
      onClick();
    },
    [onClick]
  );

  return (
    <Tooltip content={label}>
      <IconButton aria-label={label} color="fg.muted" size="2xs" variant="ghost" onClick={handleClick}>
        <Icon as={RotateCcwIcon} boxSize="2.5" />
      </IconButton>
    </Tooltip>
  );
};
