import { HStack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { useTranslation } from 'react-i18next';

export interface ListPagerProps {
  /** 1-based page number shown between the controls. */
  page: number;
  hasPrevious: boolean;
  hasNext: boolean;
  /** Keeps both controls inert while a page loads. */
  isBusy?: boolean;
  onPrevious: () => void;
  onNext: () => void;
}

/**
 * Previous/next footer for server-paged lists. `aria-disabled`, not `disabled`: a disabled button drops the
 * keyboard focus it holds, and paging is usually driven from these buttons.
 */
export const ListPager = ({ hasNext, hasPrevious, isBusy = false, page, onNext, onPrevious }: ListPagerProps) => {
  const { t } = useTranslation();
  const previousUnavailable = !hasPrevious || isBusy;
  const nextUnavailable = !hasNext || isBusy;

  return (
    <HStack borderColor="border.subtle" borderTopWidth="1px" flexShrink={0} justify="center" minH="8">
      <Button
        aria-disabled={previousUnavailable}
        size="sm"
        variant="ghost"
        onClick={previousUnavailable ? undefined : onPrevious}
      >
        {t('common.previousPage')}
      </Button>
      <Text aria-live="polite" color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums">
        {t('common.pageNumber', { page })}
      </Text>
      <Button aria-disabled={nextUnavailable} size="sm" variant="ghost" onClick={nextUnavailable ? undefined : onNext}>
        {t('common.nextPage')}
      </Button>
    </HStack>
  );
};
