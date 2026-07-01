import { Box, HStack, Input, InputGroup, Text } from '@chakra-ui/react';
import { getPasswordStrength, type PasswordStrength } from '@workbench/auth/schemas';
import { IconButton } from '@workbench/components/ui';
import { EyeIcon, EyeOffIcon } from 'lucide-react';
import { useCallback, useMemo, useState, type ComponentProps } from 'react';
import { useTranslation } from 'react-i18next';

type InputProps = ComponentProps<typeof Input>;

/** Password input with an inline visibility toggle. */
export const PasswordInput = (props: InputProps) => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(false);
  const handleToggleVisibility = useCallback(() => setIsVisible((current) => !current), []);
  const endElement = useMemo(
    () => (
      <IconButton
        aria-label={isVisible ? t('auth.hidePassword') : t('auth.showPassword')}
        color="fg.muted"
        size="2xs"
        variant="ghost"
        onClick={handleToggleVisibility}
      >
        {isVisible ? <EyeOffIcon /> : <EyeIcon />}
      </IconButton>
    ),
    [handleToggleVisibility, isVisible, t]
  );

  return (
    <InputGroup endElement={endElement}>
      <Input type={isVisible ? 'text' : 'password'} {...props} />
    </InputGroup>
  );
};

const STRENGTH_META: Record<PasswordStrength, { filledSegments: number; labelKey: string; tone: string }> = {
  moderate: { filledSegments: 2, labelKey: 'auth.passwordStrength.moderate', tone: 'fg.warning' },
  strong: { filledSegments: 3, labelKey: 'auth.passwordStrength.strong', tone: 'fg.success' },
  weak: { filledSegments: 1, labelKey: 'auth.passwordStrength.weak', tone: 'fg.error' },
};

/** Three-segment strength readout; renders nothing until the user types. */
export const PasswordStrengthMeter = ({ password }: { password: string }) => {
  const { t } = useTranslation();

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
        {t(meta.labelKey)}
      </Text>
    </HStack>
  );
};
