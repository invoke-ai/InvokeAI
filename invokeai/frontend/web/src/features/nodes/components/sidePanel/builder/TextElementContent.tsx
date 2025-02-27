import type { SystemStyleObject, TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const textSx: SystemStyleObject = {
  fontSize: 'md',
  overflowWrap: 'anywhere',
  '&[data-is-empty="true"]': {
    opacity: 0.3,
  },
};

export const TextElementContent = memo(({ content, ...rest }: { content: string } & TextProps) => {
  const { t } = useTranslation();
  return (
    <Text sx={textSx} data-is-empty={content === ''} {...rest}>
      {content || t('workflows.builder.textPlaceholder')}
    </Text>
  );
});

TextElementContent.displayName = 'TextElementContent';
