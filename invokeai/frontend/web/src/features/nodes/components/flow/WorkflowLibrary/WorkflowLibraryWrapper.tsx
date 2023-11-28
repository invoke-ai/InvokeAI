import { memo, useState } from 'react';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import WorkflowLibraryContent from './WorkflowLibraryContent';
import { WorkflowCategory } from './types';

const PER_PAGE = 10;

const WorkflowLibraryWrapper = () => {
  const [page, setPage] = useState(0);
  const [category, setCategory] = useState<WorkflowCategory>('user');
  const { data } = useListWorkflowsQuery({
    page,
    per_page: PER_PAGE,
  });

  if (!data) {
    return null;
  }

  return (
    <WorkflowLibraryContent
      data={data}
      page={page}
      setPage={setPage}
      category={category}
      setCategory={setCategory}
    />
  );
};

export default memo(WorkflowLibraryWrapper);
