import type { HeadingProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const headingSx: SystemStyleObject = {
  fontWeight: 'bold',
  fontSize: '2xl',
  '&[data-is-empty="true"]': {
    opacity: 0.3,
  },
};

export const HeadingElementContent = memo(({ content, ...rest }: { content: string } & HeadingProps) => {
  const { t } = useTranslation();
  return (
    <Text sx={headingSx} data-is-empty={content === ''} {...rest}>
      {content || t('workflows.builder.headingPlaceholder')}
    </Text>
  );
});

HeadingElementContent.displayName = 'HeadingElementContent';
