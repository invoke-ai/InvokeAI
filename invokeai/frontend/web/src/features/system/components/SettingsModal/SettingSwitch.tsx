import { Badge, BadgeProps, Flex, Text, TextProps } from '@chakra-ui/react';
import IAISwitch, { IAISwitchProps } from 'common/components/IAISwitch';
import { useTranslation } from 'react-i18next';

type SettingSwitchProps = IAISwitchProps & {
  label: string;
  useBadge?: boolean;
  badgeLabel?: string;
  textProps?: TextProps;
  badgeProps?: BadgeProps;
};

export default function SettingSwitch(props: SettingSwitchProps) {
  const { t } = useTranslation();

  const {
    label,
    textProps,
    useBadge = false,
    badgeLabel = t('settings.experimental'),
    badgeProps,
    ...rest
  } = props;

  return (
    <Flex justifyContent="space-between" py={1}>
      <Flex gap={2} alignItems="center">
        <Text
          sx={{
            fontSize: 14,
            _dark: {
              color: 'base.300',
            },
          }}
          {...textProps}
        >
          {label}
        </Text>
        {useBadge && (
          <Badge
            size="xs"
            sx={{
              px: 2,
              color: 'base.700',
              bg: 'accent.200',
              _dark: { bg: 'accent.500', color: 'base.200' },
            }}
            {...badgeProps}
          >
            {badgeLabel}
          </Badge>
        )}
      </Flex>
      <IAISwitch {...rest} />
    </Flex>
  );
}
