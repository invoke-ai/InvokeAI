import { Button, Flex, IconButton } from '@invoke-ai/ui-library';
import type { Dispatch, SetStateAction } from 'react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretLeftBold, PiCaretRightBold } from 'react-icons/pi';
import type { paths } from 'services/api/schema';

const PAGES_TO_DISPLAY = 5;

type PageData = {
  page: number;
  onClick: () => void;
};

type Props = {
  page: number;
  setPage: Dispatch<SetStateAction<number>>;
  data: paths['/api/v1/workflows/']['get']['responses']['200']['content']['application/json'];
};

// kent and devon want to make this infinite scroll
export const WorkflowLibraryPagination = ({ page, setPage, data }: Props) => {
  const { t } = useTranslation();

  const handlePrevPage = useCallback(() => {
    setPage((p) => Math.max(p - 1, 0));
  }, [setPage]);

  const handleNextPage = useCallback(() => {
    setPage((p) => Math.min(p + 1, data.pages - 1));
  }, [data.pages, setPage]);

  const pages: PageData[] = useMemo(() => {
    const pages = [];
    let first = data.pages > PAGES_TO_DISPLAY ? Math.max(0, page - Math.floor(PAGES_TO_DISPLAY / 2)) : 0;
    const last = data.pages > PAGES_TO_DISPLAY ? Math.min(data.pages, first + PAGES_TO_DISPLAY) : data.pages;
    if (last - first < PAGES_TO_DISPLAY && data.pages > PAGES_TO_DISPLAY) {
      first = last - PAGES_TO_DISPLAY;
    }
    for (let i = first; i < last; i++) {
      pages.push({
        page: i,
        onClick: () => setPage(i),
      });
    }
    return pages;
  }, [data.pages, page, setPage]);

  return (
    <Flex justifyContent="center" alignItems="center" w="full" gap={1} pt={2}>
      <IconButton
        size="sm"
        variant="ghost"
        onClick={handlePrevPage}
        isDisabled={page === 0}
        aria-label={t('common.prevPage')}
        icon={<PiCaretLeftBold />}
      />

      {pages.map((p) => (
        <Button
          size="sm"
          variant={p.page === page ? 'solid' : 'outline'}
          isDisabled={data.pages === 1}
          onClick={p.page === page ? undefined : p.onClick}
          key={p.page}
          transitionDuration="0s" // the delay in animation looks jank
        >
          {p.page + 1}
        </Button>
      ))}
      <IconButton
        size="sm"
        variant="ghost"
        onClick={handleNextPage}
        isDisabled={page === data.pages - 1}
        aria-label={t('common.nextPage')}
        icon={<PiCaretRightBold />}
      />
    </Flex>
  );
};
