import type { Ref } from 'react';

import { chakra, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { PlusIcon, XIcon } from 'lucide-react';

interface AddNegativePromptButtonProps {
  'aria-describedby'?: string;
  'aria-label': string;
  disabled?: boolean;
  /** The visible copy, which the accessible name extends. */
  label: string;
  /** Text a turned-off prompt keeps, shown after the label so the retained prompt stays discoverable. */
  preview?: string;
  ref?: Ref<HTMLButtonElement>;
  onClick: () => void;
}

/** Stands where an absent negative prompt would be; activating it brings the prompt in. */
export const AddNegativePromptButton = ({
  'aria-describedby': ariaDescribedBy,
  'aria-label': ariaLabel,
  disabled,
  label,
  onClick,
  preview,
  ref,
}: AddNegativePromptButtonProps) => (
  <Button
    ref={ref}
    aria-describedby={ariaDescribedBy}
    aria-label={ariaLabel}
    color="fg.muted"
    disabled={disabled}
    justifyContent="flex-start"
    px="2"
    variant="outline"
    onClick={onClick}
  >
    <PlusIcon />
    {/* On one baseline: the preview is smaller than the label, so centring their boxes would leave it high. */}
    <chakra.span alignItems="baseline" display="flex" gap="2" minW="0">
      <span>{label}</span>
      {preview && preview.trim() !== '' ? (
        <Text as="span" color="fg.subtle" fontSize="xs" fontWeight="normal" minW="0" truncate>
          {preview}
        </Text>
      ) : null}
    </chakra.span>
  </Button>
);

interface RemoveNegativePromptButtonProps {
  disabled?: boolean;
  /** Names the control and its tooltip. */
  label: string;
  onClick: () => void;
}

/** Sits at the end of the negative prompt's label row and takes the prompt back out. */
export const RemoveNegativePromptButton = ({ disabled, label, onClick }: RemoveNegativePromptButtonProps) => (
  <IconButton
    aria-label={label}
    color="fg.muted"
    disabled={disabled}
    size="sm"
    tooltip={label}
    variant="ghost"
    onClick={onClick}
  >
    <XIcon />
  </IconButton>
);
