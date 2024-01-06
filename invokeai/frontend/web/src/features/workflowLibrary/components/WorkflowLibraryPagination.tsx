import { InvButton } from 'common/components/InvButton/InvButton';
import { InvButtonGroup } from 'common/components/InvButtonGroup/InvButtonGroup';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import type { Dispatch, SetStateAction } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import type { paths } from 'services/api/schema';

const PAGES_TO_DISPLAY = 7;

type PageData = {
  page: number;
  onClick: () => void;
};

type Props = {
  page: number;
  setPage: Dispatch<SetStateAction<number>>;
  data: paths['/api/v1/workflows/']['get']['responses']['200']['content']['application/json'];
};

const WorkflowLibraryPagination = ({ page, setPage, data }: Props) => {
  const { t } = useTranslation();

  const handlePrevPage = useCallback(() => {
    setPage((p) => Math.max(p - 1, 0));
  }, [setPage]);

  const handleNextPage = useCallback(() => {
    setPage((p) => Math.min(p + 1, data.pages - 1));
  }, [data.pages, setPage]);

  const pages: PageData[] = useMemo(() => {
    const pages = [];
    let first =
      data.pages > PAGES_TO_DISPLAY
        ? Math.max(0, page - Math.floor(PAGES_TO_DISPLAY / 2))
        : 0;
    const last =
      data.pages > PAGES_TO_DISPLAY
        ? Math.min(data.pages, first + PAGES_TO_DISPLAY)
        : data.pages;
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
    <InvButtonGroup>
      <InvIconButton
        variant="ghost"
        onClick={handlePrevPage}
        isDisabled={page === 0}
        aria-label={t('common.prevPage')}
        icon={<FaChevronLeft />}
      />
      {pages.map((p) => (
        <InvButton
          w={10}
          isDisabled={data.pages === 1}
          onClick={p.page === page ? undefined : p.onClick}
          variant={p.page === page ? 'invokeAI' : 'ghost'}
          key={p.page}
          transitionDuration="0s" // the delay in animation looks jank
        >
          {p.page + 1}
        </InvButton>
      ))}
      <InvIconButton
        variant="ghost"
        onClick={handleNextPage}
        isDisabled={page === data.pages - 1}
        aria-label={t('common.nextPage')}
        icon={<FaChevronRight />}
      />
    </InvButtonGroup>
  );
};

export default memo(WorkflowLibraryPagination);
