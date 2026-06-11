import { Box, HStack, Input, InputGroup, Text } from '@chakra-ui/react';
import { useState, type ComponentProps } from 'react';
import { EyeIcon, EyeOffIcon } from 'lucide-react';

import { getPasswordStrength, type PasswordStrength } from '../schemas';
import { IconButton } from '../../components/ui/Button';

type InputProps = ComponentProps<typeof Input>;

/** Password input with an inline visibility toggle. */
export const PasswordInput = (props: InputProps) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <InputGroup
      endElement={
        <IconButton
          aria-label={isVisible ? 'Hide password' : 'Show password'}
          color="fg.muted"
          size="2xs"
          variant="ghost"
          onClick={() => setIsVisible((current) => !current)}
        >
          {isVisible ? <EyeOffIcon /> : <EyeIcon />}
        </IconButton>
      }
    >
      <Input type={isVisible ? 'text' : 'password'} {...props} />
    </InputGroup>
  );
};

const STRENGTH_META: Record<PasswordStrength, { filledSegments: number; label: string; tone: string }> = {
  moderate: { filledSegments: 2, label: 'Moderate', tone: 'orange.400' },
  strong: { filledSegments: 3, label: 'Strong', tone: 'green.400' },
  weak: { filledSegments: 1, label: 'Weak', tone: 'red.400' },
};

/** Three-segment strength readout; renders nothing until the user types. */
export const PasswordStrengthMeter = ({ password }: { password: string }) => {
  if (!password) {
    return null;
  }

  const meta = STRENGTH_META[getPasswordStrength(password)];

  return (
    <HStack gap="2">
      <HStack flex="1" gap="1">
        {[0, 1, 2].map((segment) => (
          <Box
            key={segment}
            bg={segment < meta.filledSegments ? meta.tone : 'border.subtle'}
            flex="1"
            h="1"
            rounded="full"
          />
        ))}
      </HStack>
      <Text color={meta.tone} fontSize="2xs" fontWeight="600">
        {meta.label}
      </Text>
    </HStack>
  );
};
